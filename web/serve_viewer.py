#!/usr/bin/env python3
"""static server for the comparison viewer (data/viewer/).

  python3 web/serve_viewer.py <root> [port]            # http://127.0.0.1:8790  (tailscale serve, benches)
  python3 web/serve_viewer.py <root> [port] --https    # https://<lan-ip>:<port> for a Quest immersive session

`--https` binds every interface and uses a self-signed cert (web/.certs/) whose SAN lists this machine's LAN
addresses. The headset has to accept the warning once; WebXR refuses a plain http:// page that is not localhost.
Never use `--https` on the build host (it refuses there): the build host serves only on 127.0.0.1 behind `tailscale serve` (web/README.md).
Every response carries `Cache-Control: no-cache` (revalidate via Last-Modified -> 304), so repointing the manifest or
replacing data files shows up on reload. Directory listings are off; /scenes.json lists the scene*.json files (newest first)
for the player's scene picker.
"""
import functools
import http.server
import json
import socket
import ssl
import sys
from pathlib import Path


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, '.js': 'text/javascript', '.json': 'application/json',
                      '.csgs': 'application/octet-stream', '.splq': 'application/octet-stream', '.webp': 'image/webp'}

    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('X-Content-Type-Options', 'nosniff')
        super().end_headers()

    def list_directory(self, path):
        self.send_error(404)
        return None

    def do_GET(self):
        if self.path.split('?', 1)[0] != '/scenes.json':
            return super().do_GET()
        files = sorted(Path(self.directory).glob('scene*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        body = json.dumps([p.name for p in files]).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        if args and str(args[1])[:1] in '45':      # errors only
            super().log_message(fmt, *args)


def lan_ips() -> list[str]:
    found: list[str] = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        found.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for res in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = res[4][0]
            if ip not in found:
                found.append(ip)
    except OSError:
        pass
    return [ip for ip in found if not ip.startswith('127.') and not ip.startswith('169.254.')]


def _cert_covers(cert: Path, ips: list[str]) -> bool:
    try:
        from cryptography import x509
        loaded = x509.load_pem_x509_certificate(cert.read_bytes())
        san = loaded.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        have = {str(ip) for ip in san.get_values_for_type(x509.IPAddress)}
    except Exception:
        return False
    return all(ip in have for ip in ips)


def _write_cert(cert: Path, key: Path, ips: list[str]) -> None:
    import datetime
    import ipaddress
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    alt = [x509.DNSName('localhost'), x509.DNSName(socket.gethostname()), x509.IPAddress(ipaddress.ip_address('127.0.0.1'))]
    alt += [x509.IPAddress(ipaddress.ip_address(ip)) for ip in ips]
    now = datetime.datetime.now(datetime.timezone.utc)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, ips[0] if ips else 'localhost')])
    signed = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(private.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=1))
        .not_valid_after(now + datetime.timedelta(days=825))
        .add_extension(x509.SubjectAlternativeName(alt), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(private, hashes.SHA256())
 )
    key.write_bytes(private.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL, serialization.NoEncryption()))
    cert.write_bytes(signed.public_bytes(serialization.Encoding.PEM))


def ensure_cert(ips: list[str]) -> tuple[Path, Path]:
    cert_dir = Path(__file__).resolve().parent / '.certs'
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert, key = cert_dir / 'dev.pem', cert_dir / 'dev-key.pem'
    if not (cert.exists() and key.exists() and _cert_covers(cert, ips)):
        _write_cert(cert, key, ips)
    return cert, key


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    https = '--https' in sys.argv[1:]
    if not args:
        sys.exit('usage: serve_viewer.py <root> [port] [--https]')
    root = args[0]
    if https and socket.gethostname().split('.')[0] == 'the build host':   # the build host serves only via 127.0.0.1 + tailscale serve
        sys.exit('--https binds every interface: never on the build host (use the default 127.0.0.1 server behind tailscale serve)')
    port = int(args[1]) if len(args) > 1 else (8443 if https else 8790)
    host = '0.0.0.0' if https else '127.0.0.1'
    srv = http.server.ThreadingHTTPServer((host, port), functools.partial(Handler, directory=root))
    ips = lan_ips()
    if https:
        cert, key = ensure_cert(ips)
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_cert_chain(cert, key)
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
        shown = ips or ['127.0.0.1']
        print(f'serving {root}', flush=True)
        for ip in shown:
            print(f'  https://{ip}:{port}/', flush=True)
        print('self-signed certificate: accept the browser warning once', flush=True)
    else:
        print(f'serving {root} on http://127.0.0.1:{port}/', flush=True)
    srv.serve_forever()


if __name__ == '__main__':
    main()
