#!/usr/bin/env python3
"""
NDVideo: a video as if flown with an ND filter, a long exposure on every frame.

Each frame is a long exposure as sharpframe.py's motion_blur makes one: the
real frames of the exposure around it, each smeared along its own motion over
its slice (sharpframe.streak), the full frame kept where nothing moved
(sharpframe.blend). The motion and the smear are found once per frame for the
whole clip (the mean of its flow to the next frame and from the previous one),
so neighbouring outputs share their frames and vectors and the streaks change
smoothly; sharpframe's exposure()
built it afresh for each still, from a different subset of frames each time,
and the streaks jittered 3-6x more than a plain average of the frames. No
steadying: the frames as shot, as a sensor would add them.

    python utils/ndvideo.py clip.mp4                 # 180 degrees: half the frame interval
    python utils/ndvideo.py clip.mp4 270             # the shutter angle, degrees
    python utils/ndvideo.py clip.mp4 1/4             # or the exposure, seconds
    python utils/ndvideo.py clip.mp4 1/4 --start 0:20 --end 0:30
    python utils/ndvideo.py clip.mp4 1/4 --width 1280     # the output (and the work) at 1280 wide
    python utils/ndvideo.py clip.mp4 1/8 --fps 25         # every 2nd frame of a 50 fps clip

Writes <stem>_nd<shutter>.mp4 next to the video, audio copied.
"""
import argparse
import os
import sys
import time
from fractions import Fraction
from concurrent.futures import ThreadPoolExecutor

import av
import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import sharpframe as sf  # noqa: E402

SWS_CS = {'bt709': 'ITU709', 'bt601': 'ITU601', 'bt2020': 'BT2020'}


def parse_shutter(s):
    """'180' (degrees) | '1/4', '0.25s' (seconds) -> (value, is_seconds)."""
    s = s.strip().lower()
    try:
        if '/' in s:
            n, d = s.rstrip('s').split('/')
            v, sec = float(n) / float(d), True
        elif s.endswith('s'):
            v, sec = float(s[:-1]), True
        else:
            v, sec = float(s), False
    except ValueError:
        raise argparse.ArgumentTypeError('bad shutter %r (use 180 for degrees, 1/4 for seconds)' % s)
    if v <= 0:
        raise argparse.ArgumentTypeError('the shutter must be above 0')
    return v, sec


def shutter_label(v, sec):
    """'nd180' | 'nd1-4s' | 'nd2s' for the file name."""
    if not sec:
        return 'nd%g' % round(v, 1)
    return 'nd1-%ds' % round(1 / v) if v < 0.95 else 'nd%gs' % round(v, 2)


ENCODER_OPTIONS = {
    'libx264': lambda q: {'crf': str(q), 'preset': 'medium'},
    'libx265': lambda q: {'crf': str(q), 'preset': 'medium', 'x265-params': 'log-level=error'},
    'h264_nvenc': lambda q: {'rc': 'vbr', 'cq': str(q), 'preset': 'p5'},
    'hevc_nvenc': lambda q: {'rc': 'vbr', 'cq': str(q), 'preset': 'p5'},
}


def open_output(path, vin, size, rate, encoder, quality, full):
    """(container, video stream); the source's pixel format if the encoder takes it, else the nearest."""
    cc = vin.codec_context
    out = av.open(path, 'w')
    vs = out.add_stream(encoder, rate=rate)
    vs.width, vs.height = size
    want = cc.pix_fmt.replace('yuvj', 'yuv')
    takes = [f.name for f in vs.codec_context.codec.video_formats or []]
    deep = want.endswith('le')
    cand = [want] + (['yuv420p10le', 'p010le'] if deep else []) + ['yuv420p']
    vs.pix_fmt = next((f for f in cand if not takes or f in takes), want)
    for k in ('color_primaries', 'color_trc', 'colorspace'):
        setattr(vs.codec_context, k, getattr(cc, k))
    vs.codec_context.color_range = 2 if full else 1
    vs.codec_context.bit_rate = 0       # quality-driven, not libavcodec's 200 kbit/s default
    vs.options = ENCODER_OPTIONS.get(encoder, lambda q: {})(quality)
    if 'hevc' in encoder or '265' in encoder:
        vs.codec_tag = 'hvc1'           # Apple players want hvc1 in an .mp4
    return out, vs


def copy_audio(video, out, t0, t1):
    """The first audio stream's packets within [t0, t1), shifted to start at t0.
    Written before the video: the muxer then holds back these few MB, not the video."""
    c = av.open(video)
    if not c.streams.audio:
        c.close()
        return
    ain = c.streams.audio[0]
    aout = out.add_stream_from_template(ain)
    shift = int(round(t0 / ain.time_base))
    if t0 > 0:
        c.seek(int(t0 / ain.time_base), stream=ain)
    for pkt in c.demux(ain):
        if pkt.dts is None or pkt.pts is None:
            continue
        t = float(pkt.pts * ain.time_base)
        if t < t0:
            continue
        if t1 is not None and t >= t1:
            break
        pkt.pts -= shift
        pkt.dts -= shift
        pkt.stream = aout
        out.mux(pkt)
    c.close()


def frames_of(video, first, last, fps, w, h, color, lut, size):
    """(n, uint16 RGB at `size`) for first <= n <= last, through sharpframe.decode
    ~1 GB of frames at a time (2 GB ran out of memory beside another job); stops
    where the clip ends. Each chunk decodes from the keyframe before it (DJI: one
    a second): at 6 frames 4K decoded ~5x the frames it kept, at ~30 under 2x.
    The next chunk decodes in the background while this one is worked on: 4K ->
    1280 takes ~0.25 s a frame, as long as the rest of the work."""
    k = int(np.clip(1e9 // max(w * h * 3, size[0] * size[1] * 6), 8, 250))     # 4:2:0 planes / RGB out
    chunks = [range(a, min(a + k, last + 1)) for a in range(first, last + 1, k)]
    read = lambda want: sf.decode(video, want, fps, w, h, color, lut, None if size == (w, h) else size)
    with ThreadPoolExecutor(1) as ex:
        ahead = ex.submit(read, chunks[0]) if chunks else None
        for i, want in enumerate(chunks):
            got = ahead.result()
            ahead = ex.submit(read, chunks[i + 1]) if i + 1 < len(chunks) and len(got) == len(want) else None
            for n in sorted(got):
                yield n, got[n]
            if len(got) < len(want):
                return


def velocity(L, m, fps):
    """Frame m's motion, px/s on the grid of L: the mean of its flow to the next
    frame and the reversed one to the previous; one of them at the clip's ends."""
    fw = sf.flow_to(L[m], L[m + 1]) if m + 1 in L else None
    bw = sf.flow_to(L[m], L[m - 1]) if m - 1 in L else None
    if fw is None and bw is None:
        return None
    if fw is None:
        return -bw * fps
    return (fw if bw is None else (fw - bw) / 2) * fps


def halves(A, V, fps):
    """(before, after, length): frame A smeared along V (px/s) over the half
    frame before its time and the half after, each its own average, and the
    streak length (px) of one half. Interior frames' slices do not depend on the
    reference, so each frame is smeared once and its halves serve all the
    exposures it is in: 13 smears a frame -> 1 at 1/4 s and 50 fps."""
    h = .5 / fps
    speed = np.hypot(V[..., 0], V[..., 1])
    steps = int(np.clip(np.percentile(speed, 99) * h / 1.5, 1, 32))
    ts = [h * (i + .5) / steps for i in range(steps)]
    return sf.streak(A, V, [-t for t in ts]), sf.streak(A, V, ts), speed * h


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0].strip(),
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument('video')
    ap.add_argument('shutter', nargs='?', type=parse_shutter, default=parse_shutter('180'),
                    help='the shutter angle in degrees (180 = half a frame, 360 = a whole one) or the '
                         'exposure in seconds (1/4, 0.25s); default 180')
    ap.add_argument('-o', '--output', help='the output .mp4 (default: <stem>_nd<shutter>.mp4 next to the video)')
    ap.add_argument('--start', type=sf.parse_time, default=0., help='from this time (default: the start)')
    ap.add_argument('--end', type=sf.parse_time, help='up to this time (default: the end)')
    ap.add_argument('--when', choices=list(sf.WHEN), default='around',
                    help='the exposure around each frame (default), before it or after it')
    ap.add_argument('--fps', type=float, help='the output at this frame rate, every 2nd frame of 50 at 25; the '
                                             'exposure still made of all the source frames (default: as the source)')
    ap.add_argument('--width', type=int, help='the output this wide, the height to scale; the work is done at that size '
                                             '(default: as the source)')
    ap.add_argument('--lut-file', default='', help='.cube colour LUT applied at decode (default: none)')
    ap.add_argument('--encoder', help='libx265, libx264, hevc_nvenc, h264_nvenc ... '
                                      '(default: libx264 for an H.264 source, libx265 otherwise)')
    ap.add_argument('--quality', type=int, default=18,
                    help='crf for x264/x265, cq for nvenc: lower is better and bigger (default 18)')
    ap.add_argument('--no-audio', action='store_true', help='leave the audio out')
    ap.add_argument('--verbose', action='store_true', help="sharpframe's log for every frame")
    a = ap.parse_args()
    if not os.path.isfile(a.video):
        ap.error('video not found: %s' % a.video)

    fps, w, h, dur, color = sf.probe(a.video)
    matrix, full, pix = color
    if a.lut_file:
        full = False        # the LUT's RGB goes out as ordinary limited-range video
    ow = min(a.width or w, w)
    size = (ow, int(round(h * ow / w / 2)) * 2)
    v, sec = a.shutter
    seconds = v if sec else v / 360 / fps
    before, after = (x * seconds for x in sf.WHEN[a.when])
    nb, na = int(round(before * fps)), int(round(after * fps))
    n0 = int(round(a.start * fps))
    n1 = int(round((min(a.end, dur) if a.end and dur else (a.end or dur)) * fps))
    log = print if a.verbose else (lambda *x, **k: None)

    c = av.open(a.video)
    vin = c.streams.video[0]
    encoder = a.encoder or ('libx264' if vin.codec_context.name == 'h264' else 'libx265')
    stem = os.path.splitext(os.path.basename(a.video))[0]
    path = a.output or os.path.join(os.path.dirname(os.path.abspath(a.video)),
                                    '%s_%s.mp4' % (stem, shutter_label(v, sec)))
    rate = vin.average_rate or vin.guessed_rate
    if a.fps and a.fps < fps:
        rate = Fraction(a.fps).limit_denominator(1001)
    # the source frame at each output time
    step = fps / float(rate)
    outs = [n for n in (n0 + int(round(k * step)) for k in range(int((n1 - n0) / step) + 1)) if n < n1]
    pts_of = {n: k for k, n in enumerate(outs)}
    print('%s: %dx%d %s -> %dx%d, %.3f -> %.3f fps; exposure %.4f s (%.1f frames, %s); frames %d-%d; %s -> %s'
          % (stem, w, h, pix, size[0], size[1], fps, float(rate), seconds, seconds * fps, a.when, n0, n1 - 1, encoder, path))
    out, vs = open_output(path, vin, size, rate, encoder, a.quality, full)
    c.close()
    if not a.no_audio:
        copy_audio(a.video, out, n0 / fps, n1 / fps)

    cs = SWS_CS.get(matrix, 'ITU709')
    rng = 'JPEG' if full else 'MPEG'
    tb = 1 / rate
    started = time.time()
    total = len(outs)
    done = 0
    refs, half, L, P = {}, {}, {}, {}   # by frame: (reference, its half size); half size; luma; halves()

    def emit(n):
        nonlocal done
        ref, small = refs.pop(n)
        win = [m for m in range(n - nb, n + na + 1) if m in P]
        img = ref.astype(np.float32)
        if len(win) > 1:
            # the half slices from the first frame's time to the last's, each 1/(2 * frames between) of the exposure
            a_, b_ = win[0], win[-1]
            streaks = P[a_][1] + P[b_][0] + sum(P[m][0] + P[m][1] for m in win[1:-1])
            length = P[a_][2] + P[b_][2] + 2 * sum(P[m][2] for m in win[1:-1])
            img = sf.blend(img, small, streaks / (2 * (b_ - a_)), length)
            log('    %d frames, streaks up to %.0f px' % (len(win), np.percentile(length, 99) * size[0] / small.shape[1]))
        rgb = av.VideoFrame.from_ndarray(np.clip(img + .5, 0, 65535).astype(np.uint16), format='rgb48le')
        f = rgb.reformat(format=vs.pix_fmt, src_colorspace=cs, dst_colorspace=cs,
                         src_color_range='JPEG', dst_color_range=rng)
        f.pts, f.time_base = pts_of[n], tb
        for p in vs.encode(f):
            out.mux(p)
        done += 1
        el = time.time() - started
        left = el / done * (total - done)
        print('\r  %d/%d frames, %.2f s a frame, about %d:%02d left   '
              % (done, total, el / done, left // 60, left % 60), end='', flush=True)

    # from the frame before the first exposure (its flow) to the one after the last
    nxt, j = n0, None
    for j, rgb in frames_of(a.video, max(0, n0 - nb - 1), n1 + na, fps, w, h, color, a.lut_file, size):
        hs = cv2.resize(rgb, (size[0] // 2, size[1] // 2), interpolation=cv2.INTER_AREA)
        L[j], half[j] = sf.luma8(hs), hs.astype(np.float32)
        if j in pts_of:
            refs[j] = (rgb, half[j])
        if j - 1 >= n0 - nb and j - 1 in L:
            P[j - 1] = halves(half.pop(j - 1), velocity(L, j - 1, fps), fps)
        while nxt < n1 and nxt + na < j:        # its exposure's halves are all there
            if nxt in refs:
                emit(nxt)
            nxt += 1
            for d in (half, L, P):
                for m in [m for m in d if m < nxt - nb - 1]:
                    del d[m]
    if j is not None and j not in P and j in half:
        P[j] = halves(half.pop(j), velocity(L, j, fps), fps)
    for n in range(nxt, n1):
        if n in refs:
            emit(n)
    for p in vs.encode(None):
        out.mux(p)
    out.close()
    el = time.time() - started
    print('\ndone: %d frames in %d:%02d -> %s' % (total, el // 60, el % 60, path))


if __name__ == '__main__':
    main()
