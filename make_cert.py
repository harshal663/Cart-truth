"""Create a self-signed HTTPS cert (certs/cert.pem, certs/key.pem) valid for localhost + your LAN IP."""
import datetime
import ipaddress
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from run_server import local_ip

out = Path(__file__).resolve().parent / "certs"
out.mkdir(exist_ok=True)
ip = local_ip()

key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "carttruth.local")])
now = datetime.datetime.now(datetime.timezone.utc)
cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
        .public_key(key.public_key()).serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
            x509.IPAddress(ipaddress.ip_address(ip))]), critical=False)
        .sign(key, hashes.SHA256()))

(out / "key.pem").write_bytes(key.private_bytes(
    serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
    serialization.NoEncryption()))
(out / "cert.pem").write_bytes(cert.public_bytes(serialization.Encoding.PEM))
print(f"Created certs/cert.pem + certs/key.pem for localhost and {ip}")
print("Re-run this if your Wi-Fi / IP address changes.")
