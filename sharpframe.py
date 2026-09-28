#!/usr/bin/env python3
"""
SharpFrame: the best still a video allows at each given time, merged from the frames around it.

    python sharpframe.py clip.mp4 12.5 1:03.2 81
    python sharpframe.py job.json                  # video + times in a file
    python web.py                                  # the same in a browser page

sharpframe.json beside this file holds the default "options": any of the
command-line options below by their long name with underscores
({"sharpen_amount": 0, "lut_file": "log.cube"}). A JSON job holds
"video_file", "times" (a list of "0:12", "1:03.5", 81.2 ...) and "options"
over those. Paths in it are relative to the JSON; the command line wins.

For each time:

  1. decode the frames within --search-window-seconds around it at full bit depth
     (10-bit -> 16-bit RGB, proper 4:2:0 chroma upsampling);
  2. score them for sharpness and take the sharpest as the reference, so a
     motion-blurred or heavily compressed frame at exactly that instant is not
     the one you get;
  3. align the other sharp frames to it with dense optical flow;
  4. merge them with per-pixel weights: where a warped frame differs from the
     reference by more than the noise (motion, a flow failure) it drops out and
     the reference stays, elsewhere noise and compression blocks average away;
  5. sharpen.

Writes <stem>_<time>.jpg next to the video, in <stem>_stills/.
--passthrough skips 2-5: the frame at the time as decoded, for tests.
"""
import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import av
import cv2
import numpy as np

DEFAULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sharpframe.json')
WORKERS = max(1, min(8, (os.cpu_count() or 2) // 2))


# ------------------------------------------------------------------ decoding
# PyAV, not an ffmpeg executable: the same libavcodec, in process, and nothing
# to install beside the wheel. Frames come out as the decoder's own planes.

def probe(video):
    """(fps, width, height, duration_s, color); color = (matrix, full_range, pixel format)."""
    c = av.open(video)
    try:
        s = c.streams.video[0]
    except IndexError:
        raise SystemExit('no video stream in %s' % video)
    cc = s.codec_context
    fps = float(s.average_rate or s.guessed_rate or 30)
    dur = float(s.duration * s.time_base) if s.duration else (
        c.duration / av.time_base if c.duration else None)
    # the stream header may leave colour unset; the first frame knows
    f = next(c.decode(s))
    c.close()
    # AVColorSpace: 1 bt709, 5/6 bt601, 9/10 bt2020; AVColorRange: 2 full
    matrix = {5: 'bt601', 6: 'bt601', 9: 'bt2020', 10: 'bt2020'}.get(int(f.colorspace or 0), 'bt709')
    pix = cc.pix_fmt or f.format.name
    full = pix.startswith('yuvj') or int(f.color_range or 0) == 2
    return fps, cc.width, cc.height, dur, (matrix, full, pix)


KR_KB = {'bt709': (0.2126, 0.0722), 'bt601': (0.299, 0.114), 'bt2020': (0.2627, 0.0593)}
_chroma_maps = {}


def yuv420_to_rgb16(buf, w, h, matrix, full):
    """One 10-bit 4:2:0 frame -> uint16 RGB, full range.

    Chroma is upsampled bicubically at its true position (HEVC default siting:
    co-sited with the left luma column, between two rows), like ffmpeg's
    accurate path, but here frames convert in parallel.
    """
    if (w, h) not in _chroma_maps:
        cx = np.arange(w, dtype=np.float32) / 2
        cy = (np.arange(h, dtype=np.float32) + .5) / 2 - .5
        _chroma_maps[(w, h)] = np.meshgrid(cx, cy)
    mx, my = _chroma_maps[(w, h)]
    n, q = w * h, w * h // 4

    def plane(a, hh, ww):
        return a.reshape(hh, ww).astype(np.float32)

    y = plane(buf[:n], h, w)
    u = cv2.remap(plane(buf[n:n + q], h // 2, w // 2), mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    v = cv2.remap(plane(buf[n + q:n + 2 * q], h // 2, w // 2), mx, my, cv2.INTER_CUBIC,
                  borderMode=cv2.BORDER_REPLICATE)
    # codes -> Y' [0,1], Cb/Cr [-.5,.5] -> RGB, as one affine map applied by cv2.transform
    ys, cs, y0 = (1023., 1023., 0.) if full else (876., 896., 64.)
    kr, kb = KR_KB[matrix]
    kg = 1 - kr - kb
    A = np.array([[1, 0, 2 * (1 - kr)],
                  [1, -2 * (1 - kb) * kb / kg, -2 * (1 - kr) * kr / kg],
                  [1, 2 * (1 - kb), 0]])
    Sc = np.diag([1 / ys, 1 / cs, 1 / cs])
    off = -np.array([y0 / ys, 512 / cs, 512 / cs])
    M = (65535. * np.hstack([A @ Sc, (A @ off)[:, None]])).astype(np.float32)
    M[:, 3] += .5
    rgb = cv2.transform(cv2.merge([y, u, v]), M)
    np.clip(rgb, 0, 65535, out=rgb)
    return rgb.astype(np.uint16)


def planes420(frame, w, h):
    """A 4:2:0 frame's Y, U, V as one flat uint16 buffer of 10-bit codes, the
    layout yuv420_to_rgb16 takes; 8-bit planes are shifted up to match."""
    deep = frame.format.name.startswith('yuv420p1')
    out = []
    for p, (pw, ph) in zip(frame.planes, ((w, h), (w // 2, h // 2), (w // 2, h // 2))):
        if deep:
            a = np.frombuffer(p, np.uint16, count=ph * p.line_size // 2).reshape(ph, -1)[:, :pw]
        else:
            a = np.frombuffer(p, np.uint8, count=ph * p.line_size).reshape(ph, -1)[:, :pw]
            a = a.astype(np.uint16) << 2
        out.append(a.ravel())
    return np.concatenate(out)


def filter_graph(stream, color, lut):
    """YUV -> full-range RGB48, then the .cube LUT, in libavfilter."""
    matrix, full, _ = color
    g = av.filter.Graph()
    chain = [g.add_buffer(template=stream),
             g.add('scale', 'in_color_matrix=%s:in_range=%s:out_range=full:flags=bicubic+accurate_rnd'
                   '+full_chroma_int' % (matrix, 'pc' if full else 'tv')),
             g.add('format', 'rgb48le')]
    if lut:
        # as a keyword the path goes in as an option value, drive colon and all,
        # with none of the escaping a filter string would need
        chain += [g.add('lut3d', file=os.path.abspath(lut).replace('\\', '/')),
                  g.add('format', 'rgb48le')]
    chain.append(g.add('buffersink'))
    for a, b in zip(chain, chain[1:]):
        a.link_to(b)
    g.configure()
    return g


def decode(video, want, fps, w, h, color, lut=None, size=None):
    """{frame: uint16 RGB (H, W, 3), full range} of the frames `want`, in one
    pass from the keyframe before the first; resized to `size` (w, h) if given."""
    want = set(want)
    matrix, full, pix = color
    c = av.open(video)
    s = c.streams.video[0]
    s.thread_type = 'AUTO'  # a window decodes straight through from its keyframe
    tb = s.time_base
    fast = not lut and w % 2 == 0 and h % 2 == 0 and re.fullmatch(r'yuvj?420p(10le)?', pix)
    graph = None if fast else filter_graph(s, color, lut)
    # the keyframe at or before a quarter frame early, so the first one itself is not missed
    c.seek(max(0, int((min(want) - 0.25) / fps / tb)), stream=s)
    got = {}
    for f in c.decode(s):
        if f.pts is None:
            continue
        n = int(round(float(f.pts * tb) * fps))
        if n > max(want):
            break
        if n not in want:
            continue
        if fast:
            got[n] = planes420(f, w, h)
        else:
            graph.push(f)
            got[n] = graph.pull().to_ndarray().reshape(h, w, 3)
    c.close()

    def rgb(b):
        im = yuv420_to_rgb16(b, w, h, matrix, full) if fast else b
        return cv2.resize(im, size, interpolation=cv2.INTER_AREA) if size else im

    # the conversion is the slow half, and runs in parallel off the decoder
    with ThreadPoolExecutor(WORKERS) as ex:
        return dict(zip(got, ex.map(rgb, got.values())))


def read_frames(video, first, count, fps, w, h, color, lut=None):
    """Frames first..first+count-1 as uint16 RGB (H, W, 3), full range."""
    got = decode(video, range(first, first + count), fps, w, h, color, lut)
    return [got[n] for n in sorted(got)]


# ------------------------------------------------------------------ analysis
def luma8(rgb16):
    g = cv2.cvtColor(rgb16, cv2.COLOR_RGB2GRAY)
    return (g >> 8).astype(np.uint8) if g.dtype == np.uint16 else g


def sharpness(rgb16):
    """Tenengrad on a half-size luma of the central 80%: large for crisp edges, noise mostly averaged out."""
    g = cv2.cvtColor(rgb16, cv2.COLOR_RGB2GRAY).astype(np.float32)
    h, w = g.shape
    g = g[int(h * .1):int(h * .9), int(w * .1):int(w * .9)]
    g = cv2.resize(g, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(gx * gx + gy * gy))


def flow_to(ref8, img8):
    """Dense flow F with ref(x) ~ img(x + F(x)): DIS with variational refinement.

    Patches are matched down to half resolution (finest scale 1): 4x faster than
    full resolution, and the merged still came out the same (sharpness -0.8 %,
    no visible difference at 100 %).
    """
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    dis.setFinestScale(1)
    return dis.calc(ref8, img8, None)


def warp(img, flow):
    """img resampled onto the reference grid through the flow; + validity mask."""
    h, w = img.shape[:2]
    mx, my = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx += flow[..., 0]
    my += flow[..., 1]
    out = cv2.remap(img, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    valid = ((mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)).astype(np.float32)
    return out, valid


def merge(frames, ref_i, use, strength, soft, log):
    """Weighted average of the `use` frames on the grid of frames[ref_i].

    A pixel of a warped frame counts where it matches the reference to within
    the noise: the local energy of the luma difference (blurred square, which
    half-pixel shifts of fine texture cannot cancel the way a blurred signed
    difference does) against 3x its value over the best-aligned tenth of the
    picture, where it is noise alone. Test with noise added to every frame,
    against the clean reference: one frame 33.9 dB, this 36.3-36.5.

    `soft` is the older rule, the blurred signed difference against 2.5x its
    median: it lets misaligned texture in, a smoother, dreamier picture
    (32.3 dB on that test).
    """
    ref = frames[ref_i]
    ref8 = luma8(ref)
    base = ref.astype(np.float32)
    base_g = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY)
    base_l = cv2.GaussianBlur(base_g, (0, 0), 1.5)
    acc = base.copy()
    wsum = np.ones(base.shape[:2], np.float32)
    lock = threading.Lock()
    todo = [i for i in use if i != ref_i]
    done = [0]

    def add(i):
        img, valid = warp(frames[i], flow_to(ref8, luma8(frames[i])))
        img = img.astype(np.float32)
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if soft:
            d = np.abs(cv2.GaussianBlur(g, (0, 0), 1.5) - base_l)
        else:
            d = cv2.GaussianBlur((g - base_g) ** 2, (0, 0), 1.5)
        tau = strength * max(2.5 * float(np.median(d)) if soft else 3 * float(np.percentile(d, 10)), 1.0)
        wgt = np.exp(-(d / tau) ** 2) * valid
        img *= wgt[..., None]
        with lock:
            acc[...] += img
            wsum[...] += wgt
            done[0] += 1
            log('      aligned %d/%d  (frame %+d, mean weight %.2f)' % (done[0], len(todo), i - ref_i,
                                                                      float(wgt.mean())))

    # DIS itself is single-threaded, so the frames go in parallel
    with ThreadPoolExecutor(WORKERS) as ex:
        list(ex.map(add, todo))
    return acc / wsum[..., None], float(wsum.mean())


def window(t, fps, dur, seconds):
    """(requested frame, first frame, count) of the frames around time t; None outside the clip."""
    c = int(round(t * fps))
    n_total = int(round(dur * fps)) if dur else None
    if c < 0 or n_total is not None and c >= n_total:
        return None
    r = max(0, int(round(seconds * fps)))
    first = max(0, c - r)
    last = c + r if n_total is None else min(n_total - 1, c + r)
    return c, first, last - first + 1


# ------------------------------------------------------------------ motion
# A long exposure by the reference frame, before it, around it or after it:
# what moves in the picture streaks along its way, what stands still in it
# (the point a forward flight heads for, a subject the camera follows) stays
# sharp. ~10 real frames spread over the exposure, each steadied as the
# camera should have held it, each smeared along its own motion over its
# slice of the exposure, the slices added up. Real frames, not the reference
# alone smeared: what is hidden behind a thing held sharp at the reference is
# there in the others, where the reference alone drags the thing's colour into
# the streaks round it.
PATH_SAMPLES = 10    # frames the exposure is made of, however long it is


def exposure(video, ref_n, before, after, fps, w, h, color, lut, log, focus=None):
    """The exposure's frames in time order: dicts of `t` (seconds from the
    reference), `A` (the frame, half size, float, steadied; None for the
    reference itself) and `V` (the motion there, px/s on that grid, steadied).
    [] when the clip has nothing there. `focus` (x, y as fractions of the frame)
    is held still instead of what stands still: panning after it."""
    nb = min(ref_n, max(0, int(round(before * fps))))
    na = max(0, int(round(after * fps)))
    if nb + na < 1:
        return []
    # the frames shared out between the two sides by their length
    sides = []
    for n, sign in ((nb, -1), (na, 1)):
        if n:
            k = max(1, min(n, round(PATH_SAMPLES * n / (nb + na))))
            sides.append(sorted({ref_n + sign * int(round(n * j / k)) for j in range(k + 1)},
                                key=lambda i: abs(i - ref_n)))
    size = (w // 2, h // 2)
    frames = decode(video, {i for side in sides for i in side}, fps, w, h, color, lut, size)
    L = {n: luma8(f) for n, f in frames.items()}
    gw, gh = size
    Z = np.dstack(np.meshgrid(np.arange(gw, dtype=np.float32), np.arange(gh, dtype=np.float32)))
    pos = {ref_n: Z}        # where each point of the reference is in that frame
    out = {}                # frame: (flow to the next one out, that one)
    for side in sides:
        side = [i for i in side if i in L]      # past the clip's end nothing decodes
        P, last = Z.copy(), np.zeros_like(Z)
        # chained frame to frame: from the reference straight to one a second away the flow would get lost
        for a, b in zip(side, side[1:]):
            F = flow_to(L[a], L[b])     # L[a](x) ~ L[b](x + F(x))
            out.setdefault(a, (F, b))
            step = cv2.remap(F, P[..., 0], P[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            # a point gone over the edge has no flow any more: it goes on as it went
            inside = ((P[..., 0] >= 0) & (P[..., 0] <= gw - 1) & (P[..., 1] >= 0) & (P[..., 1] <= gh - 1))[..., None]
            last = np.where(inside, step, last)
            P = P + last
            pos[b] = P.copy()
    order = sorted(pos)
    C = holding(Z, pos, focus) if focus is not None else steady(Z, pos, ref_n, fps)
    t = {n: (n - ref_n) / fps for n in order}
    V = {}
    for n in order:
        if n not in out:
            continue
        F, m = out[n]
        # a point of frame n and where it is in frame m, both in the steadied picture
        here, there = cv2.transform(Z, C[n]), cv2.transform(Z + F, C[m])
        V[n] = cv2.warpAffine((there - here) / (t[m] - t[n]), C[n], size, flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    for side in sides:          # the outermost frame of a side goes on as the one before it
        side = [i for i in side if i in pos]
        for a, b in zip(side, side[1:]):
            if b not in V and a in V:
                V[b] = V[a]
    log('    motion: %d frames, %.3f s before the frame, %.3f s after' % (len(order), nb / fps, na / fps))
    return [dict(t=t[n], V=V[n], A=None if n == ref_n else
                 cv2.warpAffine(frames[n].astype(np.float32), C[n], size, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)) for n in order if n in V]


def holding(Z, pos, focus):
    """{frame: 2x3} that keeps the focus where it is in the reference: each
    frame shifted by the focus's own way, the median over a patch ~3 % of the
    frame across (one point's flow is too jumpy)."""
    gh, gw = Z.shape[:2]
    cx, cy = int(focus[0] * (gw - 1)), int(focus[1] * (gh - 1))
    r = max(4, gw // 64)
    patch = (slice(max(0, cy - r), cy + r + 1), slice(max(0, cx - r), cx + r + 1))
    C = {}
    for n, P in pos.items():
        dx, dy = np.median((P - Z)[patch].reshape(-1, 2), axis=0)
        C[n] = np.float32([[1, 0, -dx], [0, 1, -dy]])
    return C


def steady(Z, pos, ref_n, fps):
    """{frame: 2x3} that takes the shake out: the whole picture's motion (a
    similarity fitted per frame) replaced by a quadratic in time through it, so
    the flight and the turns stay and the jitter goes. Too few frames to tell
    the one from the other: as they are."""
    I = np.float32([[1, 0, 0], [0, 1, 0]])
    others = [n for n in pos if n != ref_n]
    if len(others) < 4:
        return {n: I for n in pos}
    step = 16
    src = Z[::step, ::step].reshape(-1, 2)
    fits = {}
    for n in others:
        M, _ = cv2.estimateAffinePartial2D(src, pos[n][::step, ::step].reshape(-1, 2), method=cv2.RANSAC,
                                           ransacReprojThreshold=2.0)
        if M is None:
            return {n: I for n in pos}
        fits[n] = M
    t = np.array([(n - ref_n) / fps for n in others])
    # log scale, angle, shift: 0 at the reference itself, so the fit goes through the origin
    q = np.array([[np.log(np.hypot(M[0, 0], M[1, 0])), np.arctan2(M[1, 0], M[0, 0]), M[0, 2], M[1, 2]]
                  for M in (fits[n] for n in others)])
    A = np.stack([t, t * t], 1)
    smooth = A @ np.linalg.lstsq(A, q, rcond=None)[0]
    C = {ref_n: I}
    for n, (ls, an, tx, ty) in zip(others, smooth):
        sc = np.exp(ls)
        S = np.array([[sc * np.cos(an), -sc * np.sin(an), tx], [sc * np.sin(an), sc * np.cos(an), ty], [0, 0, 1]])
        C[n] = (S @ np.linalg.inv(np.vstack([fits[n], [0, 0, 1]])))[:2].astype(np.float32)
    return C


class Guide:
    """He et al.'s guided filter with a colour guide, the guide's part done once
    for all the fields filtered by it. Colour, not luma: an orange jacket
    against yellow leaves is one brightness. I: RGB in 0..1; eps ~ the squared
    colour difference that counts as an edge."""

    def __init__(self, I, r, eps):
        self.I, self.box = I, (lambda x: cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1)))
        self.mI = mI = self.box(I)
        v = lambda a, b: self.box(I[..., a] * I[..., b]) - mI[..., a] * mI[..., b] + (eps if a == b else 0)
        rr, rg, rb, gg, gb, bb = v(0, 0), v(0, 1), v(0, 2), v(1, 1), v(1, 2), v(2, 2)
        # the symmetric 3x3 inverse by cofactors: np.linalg.inv over 2.7M of them takes 45 s
        c = [gg * bb - gb * gb, gb * rb - rg * bb, rg * gb - gg * rb, rr * bb - rb * rb, rg * rb - rr * gb,
             rr * gg - rg * rg]
        det = rr * c[0] + rg * c[1] + rb * c[2]
        self.inv = [x / det for x in c]     # (rr, rg, rb, gg, gb, bb) of the inverse

    def __call__(self, p):
        """p (h, w) or (h, w, c) smoothed, its edges following the guide's."""
        I, box, mI = self.I, self.box, self.mI
        rr, rg, rb, gg, gb, bb = self.inv
        chans = cv2.split(p) if p.ndim == 3 else [p]
        out = []
        for ch in chans:
            mp = box(ch)
            cv = box(I * ch[..., None]) - mI * mp[..., None]
            a = np.dstack([rr * cv[..., 0] + rg * cv[..., 1] + rb * cv[..., 2],
                           rg * cv[..., 0] + gg * cv[..., 1] + gb * cv[..., 2],
                           rb * cv[..., 0] + gb * cv[..., 1] + bb * cv[..., 2]])
            b = mp - (a * mI).sum(-1)
            out.append((box(a) * I).sum(-1) + box(b))
        return cv2.merge(out) if p.ndim == 3 else out[0]


def motion_blur(img, frames, log):
    """The exposure: each frame smeared along its own motion over its slice of
    the exposure (from halfway to the one before to halfway to the one after),
    the slices added up; where hardly anything moved (< ~1 px of the full
    frame) the full-size still stays, fading into the streaks up to ~5 px
    along the picture's own edges.

    A streak's sample counts only where its source moves as the pixel does
    (the two paths within 1.5 px over the step): across an edge it comes off
    another thing, one the pixel never saw. The streaks are drawn on the
    half-size grid in steps of <= 1.5 px of it."""
    if len(frames) < 2:
        return img
    H, W = img.shape[:2]
    gh, gw = frames[0]['V'].shape[:2]
    small = cv2.resize(img, (gw, gh), interpolation=cv2.INTER_AREA)
    T = [f['t'] for f in frames]
    X, Y = np.meshgrid(np.arange(gw, dtype=np.float32), np.arange(gh, dtype=np.float32))
    total = np.zeros_like(small)
    length = np.zeros((gh, gw), np.float32)
    steps_all = 0
    for k, f in enumerate(frames):
        lo = T[0] if k == 0 else (T[k - 1] + T[k]) / 2
        hi = T[-1] if k == len(T) - 1 else (T[k] + T[k + 1]) / 2
        A = small if f['A'] is None else f['A']
        V = f['V']
        speed = np.hypot(V[..., 0], V[..., 1])
        length += speed * (hi - lo)
        steps = int(np.clip(np.percentile(speed, 99) * (hi - lo) / 1.5, 1, 32))
        steps_all += steps
        acc = np.zeros_like(A)
        wsum = np.zeros((gh, gw), np.float32)
        for i in range(steps):
            dt = lo + (hi - lo) * (i + .5) / steps - f['t']
            # what is at x then was at x - V(x) dt now; it counts if it moves as x does
            sx, sy = X - V[..., 0] * dt, Y - V[..., 1] * dt
            Vs = cv2.remap(V, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            wt = np.exp(-((np.hypot(Vs[..., 0] - V[..., 0], Vs[..., 1] - V[..., 1]) * abs(dt)) / 1.5) ** 2)
            acc += cv2.remap(A, sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) * wt[..., None]
            wsum += wt
        # where no sample held (a gap opening behind a thing) the frame itself stays
        part = np.where(wsum[..., None] > 1e-3, acc / np.maximum(wsum, 1e-3)[..., None], A)
        total += part * ((hi - lo) / (T[-1] - T[0]))
    streaks = cv2.resize(total, (W, H), interpolation=cv2.INTER_CUBIC)
    edges = Guide((small / 65535.).clip(0, 1).astype(np.float32), max(4, gw // 240), 1e-3)
    m = np.clip(edges(np.clip((length * (W / gw) - 1) / 4, 0, 1).astype(np.float32)), 0, 1)
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    log('    motion blur: streaks up to %.0f px, %d steps over %d frames' % (
        np.percentile(length, 99) * W / gw, steps_all, len(frames)))
    return img * (1 - m) + streaks * m


def still(frames, at, a, log=print, blur=None):
    """(merged still, the sharpest frame alone, info) from one window of frames,
    both sharpened; `at` is the requested frame's index in the window. `blur`,
    given, takes the merged still and the reference's index in the window
    before the sharpening: the motion stage."""
    with ThreadPoolExecutor(WORKERS) as ex:
        s = np.array(list(ex.map(sharpness, frames)))
    # the moment was chosen: a frame at the window's edge must be ~18 % sharper to win
    dist = np.abs(np.arange(len(s)) - at) / max(1, at, len(s) - 1 - at)
    ref = int(np.argmax(s * (1 - 0.15 * dist ** 2)))
    order = [i for i in np.argsort(-s) if s[i] >= a.min_relative_sharpness * s[ref]][:max(1, a.max_frames_to_merge)]
    if ref not in order:
        order = [ref] + order[:-1]
    rel = float(s[at] / s[ref]) if 0 <= at < len(s) else None
    log('    reference frame %+d of the requested, which has %s of its sharpness; merging %d frames'
        % (ref - at, '%.2f' % rel if rel is not None else '?', len(order)))
    img, depth = merge(frames, ref, order, a.merge_tolerance, a.soft, log)
    if blur:
        img = blur(img, ref)
    sharp = lambda im: unsharp(im, a.sharpen_amount, a.sharpen_radius_px)
    info = dict(ref=ref - at, rel=rel, merged=len(order), depth=depth)
    return sharp(img), sharp(frames[ref].astype(np.float32)), info


# ------------------------------------------------------------------ output
def unsharp(img, amount, radius):
    if amount <= 0:
        return img
    return img + amount * (img - cv2.GaussianBlur(img, (0, 0), radius))


def jpeg(img_f, quality=96):
    """Float RGB on the 16-bit scale -> JPEG bytes."""
    rgb8 = np.clip(img_f / 257. + .5, 0, 255).astype(np.uint8)
    ok, data = cv2.imencode('.jpg', cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise SystemExit('JPEG encoding failed')
    return data.tobytes()


def save(img_f, path_jpg):
    # cv2.imwrite cannot take a non-ASCII path on Windows (it writes a garbled name or nothing)
    with open(path_jpg, 'wb') as fh:
        fh.write(jpeg(img_f))


def parse_time(s):
    """12.5 | 1:03.2 | 0:01:03.2 -> seconds."""
    parts = s.strip().split(':')
    try:
        v = 0.0
        for p in parts:
            v = v * 60 + float(p)
        return v
    except ValueError:
        raise argparse.ArgumentTypeError('bad time %r (use 12.5, 1:03.2 or 0:01:03.2)' % s)


def time_label(t):
    return '%02dm%06.3fs' % (int(t // 60), t - 60 * int(t // 60))


def load_job(path):
    """The JSON job; a trailing comma before ] or } is forgiven, other errors point at the line."""
    with open(path, encoding='utf-8-sig') as fh:
        text = fh.read()
    # strings are matched first so a ", ]" inside one is left alone
    text = re.sub(r'("(?:\\.|[^"\\])*")|,(\s*[\]}])', lambda m: m.group(1) or m.group(2), text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        line = text.splitlines()[e.lineno - 1] if e.lineno <= len(text.splitlines()) else ''
        raise SystemExit('%s, line %d: %s\n    %s\n    %s^' % (path, e.lineno, e.msg, line, ' ' * (e.colno - 1)))


def job_options(ap, path):
    """The JSON job, its "options" made the parser's defaults; files named in it are relative to it."""
    job = load_job(path)
    extra = sorted(set(job) - {'video_file', 'times', 'options'})
    if extra:
        ap.error('%s: unknown keys %s (expected video_file, times, options)' % (path, ', '.join(extra)))
    opts = {k.replace('-', '_'): v for k, v in (job.get('options') or {}).items()}
    bad = sorted(set(opts) - {a.dest for a in ap._actions})
    if bad:
        ap.error('%s: unknown options %s' % (path, ', '.join(bad)))
    for key in ('lut_file', 'output_folder'):
        if opts.get(key):
            opts[key] = os.path.join(os.path.dirname(os.path.abspath(path)), opts[key])
    ap.set_defaults(**opts)
    return job


def parse_focus(s):
    """'0.4,0.6' -> (0.4, 0.6); '' -> None."""
    if not s:
        return None
    x, y = (float(v) for v in str(s).split(','))
    return min(max(x, 0.), 1.), min(max(y, 0.), 1.)


WHEN = {'before': (1, 0), 'around': (.5, .5), 'after': (0, 1)}   # the exposure's share before / after


def motion_stage(a, video, first, fps, w, h, color, log=print):
    """The `blur` for still(): the long exposure, if a.motion_seconds asks for one."""
    if a.motion_seconds <= 0:
        return None
    b, f = WHEN[a.motion_when]
    return lambda img, ref: motion_blur(img, exposure(video, first + ref, b * a.motion_seconds, f * a.motion_seconds,
                                                      fps, w, h, color, a.lut_file, log,
                                                      parse_focus(a.motion_focus)), log)


def motion_suffix(a):
    """'_motion1-8s' for the file name (no '/' in one), + '_around' / '_after'
    off the default, + '_focus' when held at a point; '' without the motion stage."""
    s = a.motion_seconds
    if s <= 0:
        return ''
    return ('_motion1-%ds' % round(1 / s) if s < 0.95 else '_motion%gs' % round(s, 2)) + (
        '' if a.motion_when == 'before' else '_' + a.motion_when) + ('_focus' if a.motion_focus else '')


def parser():
    """The command line, its defaults from sharpframe.json."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('video', help='the video, or a .json job with "video_file", "times" and "options"')
    ap.add_argument('times', nargs='*', type=parse_time, help='12.5, 1:03.2 or 0:01:03.2')
    ap.add_argument('--times-file', help='text file with one time per line (# comments allowed)')
    ap.add_argument('-o', '--output-folder', default='',
                    help='where the stills go (default: <video folder>/<video name>_stills)')
    ap.add_argument('--search-window-seconds', type=float, default=0.12,
                    help='frames within this many seconds on each side of a time are searched and merged '
                         '(default 0.12)')
    ap.add_argument('--max-frames-to-merge', type=int, default=9,
                    help='frames merged at most, the sharpest one included (default 9)')
    ap.add_argument('--min-relative-sharpness', type=float, default=0.75,
                    help='frames less sharp than this share of the sharpest one are not merged (default 0.75)')
    ap.add_argument('--merge-tolerance', type=float, default=1.0,
                    help='higher averages more (less noise), lower keeps more of the sharpest frame '
                         '(less blur, fewer ghosts) (default 1)')
    ap.add_argument('--soft', action='store_true',
                    help='merge more boldly: smoother and softer, a dreamy look (the older default)')
    ap.add_argument('--motion-seconds', type=float, default=0.0,
                    help='a long exposure of this many seconds, by the moment: what moves in the picture '
                         'streaks, what stands still in it stays sharp; 0 = off (default)')
    ap.add_argument('--motion-when', choices=list(WHEN), default='before',
                    help='the exposure before the moment (streaks trail behind what moves), around it, '
                         'or after it (default: before)')
    ap.add_argument('--motion-focus', default='',
                    help='x,y as fractions of the frame (0.5,0.5 = the centre) held sharp with --motion-seconds, '
                         'the rest streaking against it (default: what stands still in the picture)')
    ap.add_argument('--sharpen-amount', type=float, default=0.6, help='unsharp-mask amount, 0 = off (default 0.6)')
    ap.add_argument('--sharpen-radius-px', type=float, default=1.0,
                    help='unsharp-mask radius in pixels (default 1.0)')
    ap.add_argument('--lut-file', default='',
                    help='.cube color LUT applied at decode, e.g. a log profile -> Rec.709 (default: none)')
    ap.add_argument('--save-single-frame', action='store_true',
                    help='also save the sharpest frame alone (<name>_single_frame), processed the same way, '
                         'to compare with the merged still')
    ap.add_argument('--passthrough', action='store_true',
                    help='for tests: save the frame at the time as decoded (LUT included), no search, merge, '
                         'motion or sharpening (<name>_passthrough)')
    ap.add_argument('--open-folder', action='store_true', help='open the output folder at the end (Windows)')
    if os.path.isfile(DEFAULTS):
        job_options(ap, DEFAULTS)
    return ap


def main():
    ap = parser()
    args = ap.parse_args()
    job_times = []
    if args.video.lower().endswith('.json'):
        job = job_options(ap, args.video)
        args = ap.parse_args()      # command line over the JSON's options
        if 'video_file' not in job:
            ap.error('%s: no "video_file"' % args.video)
        args.video = os.path.join(os.path.dirname(os.path.abspath(args.video)), job['video_file'])
        job_times = [parse_time(str(x)) for x in job.get('times') or []]
    if not os.path.isfile(args.video):
        ap.error('video not found: %s' % args.video)

    times = job_times + list(args.times)
    if args.times_file:
        with open(args.times_file, encoding='utf-8') as fh:
            for line in fh:
                line = line.split('#')[0].strip()
                if line:
                    times += [parse_time(x) for x in re.split(r'[\s,;]+', line) if x]
    if not times:
        ap.error('give at least one time')

    fps, w, h, dur, color = probe(args.video)
    stem = os.path.splitext(os.path.basename(args.video))[0]
    out = args.output_folder or os.path.join(os.path.dirname(os.path.abspath(args.video)), stem + '_stills')
    os.makedirs(out, exist_ok=True)
    print('%s: %dx%d, %.3f fps, %.2f s, %s %s; %s'
          % (stem, w, h, fps, dur or 0, color[2], color[0], 'passthrough' if args.passthrough else
             'window +-%d frames' % int(round(args.search_window_seconds * fps))))

    jobs = []
    for t in times:
        wnd = window(t, fps, dur, 0 if args.passthrough else args.search_window_seconds)
        if wnd is None:
            print('%s: outside the clip, skipped' % t)
        else:
            jobs.append((t,) + wnd)

    # the next time's frames decode while the current one merges
    decoder = ThreadPoolExecutor(1)
    decode = lambda j: decoder.submit(read_frames, args.video, j[2], j[3], fps, w, h, color, args.lut_file)
    pending = decode(jobs[0]) if jobs else None
    start = time.time()
    for k, (t, c, first, _) in enumerate(jobs):
        t0 = time.time()
        if k:
            left = (time.time() - start) / k * (len(jobs) - k)
            eta = ', about %d:%02d left' % (left // 60, left % 60)
        else:
            eta = ''
        print('[%d/%d] %s%s' % (k + 1, len(jobs), time_label(t), eta), flush=True)
        frames = pending.result()
        pending = decode(jobs[k + 1]) if k + 1 < len(jobs) else None
        if not frames:
            print('%s: no frames decoded, skipped' % t)
            continue
        if args.passthrough:
            name = '%s_%s_passthrough' % (stem, time_label(t))
            save(frames[0].astype(np.float32), os.path.join(out, name + '.jpg'))
            print('    -> %s.jpg  (%.1f s)' % (name, time.time() - t0))
            continue
        img, single, info = still(frames, c - first, args, blur=motion_stage(args, args.video, first, fps, w, h, color))
        name = '%s_%s%s' % (stem, time_label(t), motion_suffix(args))
        save(img, os.path.join(out, name + '.jpg'))
        if args.save_single_frame:
            save(single, os.path.join(out, name + '_single_frame.jpg'))
        print('    -> %s.jpg  (average %.1f frames per pixel, %.1f s)' % (name, info['depth'], time.time() - t0))
    if jobs:
        el = time.time() - start
        print('done: %d stills in %d:%02d -> %s' % (len(jobs), el // 60, el % 60, out))
        if args.open_folder:
            os.startfile(out)


if __name__ == '__main__':
    main()
