#!/usr/bin/env python3
"""SharpFrame in a browser page: drop a video, find the moment, get the photo.

    python web.py [--port 8765] [--no-browser]

A local server on 127.0.0.1 only. A browser gives the page the file but never
its path, so the video is copied into a temp folder (removed on exit); the
page plays it from the file meanwhile, to find the moment.
"""
import argparse
import atexit
import glob
import http.server
import json
import os
import shutil
import sys
import tempfile
import threading
import uuid
import webbrowser
from urllib.parse import parse_qs, urlparse

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # the embedded Python of the zip leaves the script's folder off
import sharpframe as sf  # noqa: E402
TMP = tempfile.mkdtemp(prefix='sharpframe_')
atexit.register(shutil.rmtree, TMP, True)
# held open while this runs: Windows will not delete it, so sweep() leaves this folder alone
LOCK = open(os.path.join(TMP, '.lock'), 'w')
atexit.register(LOCK.close)
CHUNK = 8 << 20
KEEP = 40            # results kept in memory, ~5 MB each at 4K
QUALITY = 96         # JPEG quality of the photos, as sharpframe.py saves them

video = {}           # the current one: path, name, fps, w, h, dur, color
results = {}         # id -> JPEG bytes
work = threading.Lock()   # one still at a time: a 4K window is ~1 GB of frames


def sweep():
    """The temp folders of runs gone without cleaning up (a console window
    closed with its X skips atexit), each holding a copy of a video."""
    for d in glob.glob(os.path.join(tempfile.gettempdir(), 'sharpframe_*')):
        if d == TMP:
            continue
        try:
            os.remove(os.path.join(d, '.lock'))
        except FileNotFoundError:
            pass
        except OSError:     # its run is still on
            continue
        shutil.rmtree(d, True)


def options():
    """sharpframe.json as the command line would read it, re-read so edits apply at once;
    the page's `soft` switch goes over it."""
    return sf.parser().parse_args(['-'])


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def reply(self, code, body, ctype='application/json'):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(body)

    def fail(self, msg, code=400):
        self.reply(code, {'error': str(msg)})

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == '/':
            with open(os.path.join(HERE, 'web.html'), 'rb') as f:
                return self.reply(200, f.read(), 'text/html; charset=utf-8')
        if u.path.startswith('/img/'):
            data = results.get(u.path[5:].split('.')[0])
            return self.reply(200, data, 'image/jpeg') if data else self.fail('gone', 404)
        if u.path == '/preview':
            return self.preview(float(parse_qs(u.query).get('t', ['0'])[0]))
        self.fail('not found', 404)

    def do_PUT(self):
        if urlparse(self.path).path != '/upload':
            return self.fail('not found', 404)
        name = os.path.basename(parse_qs(urlparse(self.path).query).get('name', ['video.mp4'])[0])
        left = int(self.headers.get('Content-Length', 0))
        video.clear()
        for f in set(os.listdir(TMP)) - {'.lock'}:   # one video at a time: they run to gigabytes
            try:
                os.remove(os.path.join(TMP, f))
            except OSError:         # still open for a photo being made; goes on exit
                pass
        path = os.path.join(TMP, uuid.uuid4().hex[:8] + os.path.splitext(name)[1].lower())
        free = shutil.disk_usage(TMP).free
        err = None if free > left + (256 << 20) else 'not enough space for the copy in %s: %.1f GB free, %.1f GB needed' % (
            TMP, free / 2 ** 30, left / 2 ** 30)
        out = None if err else open(path, 'wb')
        # the body is read to the end even after an error: a reply cut in mid-upload
        # reaches the page only as a dropped connection
        while left > 0:
            b = self.rfile.read(min(CHUNK, left))
            if not b:
                err = err or 'upload cut short'
                break
            left -= len(b)
            if out:
                try:
                    out.write(b)
                except OSError as e:
                    err, out = 'cannot write the copy: %s' % e, out.close()
        if out:
            out.close()
        if err:
            print(err, flush=True)
            return self.fail(err)
        try:
            fps, w, h, dur, color = sf.probe(path)
        except (Exception, SystemExit) as e:
            return self.fail('not a video I can read: %s' % e)
        video.update(path=path, name=name, fps=fps, w=w, h=h, dur=dur, color=color)
        print('%s: %dx%d, %.3f fps, %.1f s' % (name, w, h, fps, dur or 0), flush=True)
        self.reply(200, {'fps': fps, 'w': w, 'h': h, 'dur': dur})

    def do_POST(self):
        if urlparse(self.path).path != '/still':
            return self.fail('not found', 404)
        req = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
        t = float(req.get('t', 0))
        v = dict(video)   # a new upload may clear it meanwhile
        if not v:
            return self.fail('no video yet')
        a = options()
        # the page's switches over sharpframe.json
        if 'soft' in req:
            a.soft = bool(req['soft'])
        if 'motion' in req:
            a.motion_seconds = max(0.0, float(req['motion'] or 0))
        if req.get('when') in sf.WHEN:
            a.motion_when = req['when']
        if 'focus' in req:
            a.motion_focus = '%g,%g' % tuple(req['focus']) if req['focus'] else ''
        wnd = sf.window(t, v['fps'], v['dur'], 0 if a.passthrough else a.search_window_seconds)
        if wnd is None:
            return self.fail('outside the clip')
        c, first, count = wnd
        with work:
            print('%s at %s' % (v['name'], sf.time_label(t)), flush=True)
            frames = sf.read_frames(v['path'], first, count, v['fps'], v['w'], v['h'],
                                    v['color'], a.lut_file)
            if not frames:
                return self.fail('no frames decoded there')
            at = min(c - first, len(frames) - 1)
            if a.passthrough:   # sharpframe.json's test mode: the frame as decoded
                a.soft, a.motion_seconds = False, 0
                img, info = frames[at].astype('float32'), dict(ref=0, rel=1.0, merged=1, depth=1.0)
            else:
                img, _, info = sf.still(frames, at, a, blur=sf.motion_stage(a, v['path'], first, v['fps'], v['w'],
                                                                            v['h'], v['color']))
            key = uuid.uuid4().hex[:12]
            results[key] = sf.jpeg(img, QUALITY)
            results[key + 'f'] = sf.jpeg(frames[at].astype('float32'), 92)  # as a player shows it
        for k in list(results)[:-2 * KEEP]:
            del results[k]
        stem = os.path.splitext(v['name'])[0]
        name = '%s_%s%s%s%s.jpg' % (stem, sf.time_label(t), '_soft' if a.soft else '', sf.motion_suffix(a),
                                    '_passthrough' if a.passthrough else '')
        self.reply(200, dict(info, id=key, t=t, name=name, soft=a.soft, w=v['w'], h=v['h'],
                             motion=a.motion_seconds, when=a.motion_when, focus=sf.parse_focus(a.motion_focus),
                             passthrough=a.passthrough, format='JPEG', quality=QUALITY, bytes=len(results[key])))

    def preview(self, t):
        """One frame near t, small: for a video the browser cannot play itself."""
        v = dict(video)   # a new upload may clear it meanwhile
        if not v:
            return self.fail('no video yet')
        c = min(max(0, int(round(t * v['fps']))), int((v['dur'] or 1e9) * v['fps']) - 1)
        fr = sf.read_frames(v['path'], c, 1, v['fps'], v['w'], v['h'], v['color'])
        if not fr:
            return self.fail('no frame there')
        s = 1280 / max(v['w'], v['h'])
        img = cv2.resize(fr[0], None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else fr[0]
        self.reply(200, sf.jpeg(img.astype('float32'), 85), 'image/jpeg')


def main():
    ap = argparse.ArgumentParser(description='SharpFrame in a browser page')
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--no-browser', action='store_true')
    a = ap.parse_args()
    sweep()
    for port in range(a.port, a.port + 20):
        try:
            srv = http.server.ThreadingHTTPServer(('127.0.0.1', port), Handler)
            break
        except OSError:
            continue
    else:
        raise SystemExit('no free port from %d' % a.port)
    url = 'http://127.0.0.1:%d/' % port
    print('SharpFrame: %s  (close this window to quit)' % url, flush=True)
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
