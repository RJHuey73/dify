import hashlib
from typing import Union

from Crypto.Cipher import AES
from Crypto.Hash import SHA256
from Crypto.Protocol.KDF import HKDF
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes

from configs import dify_config
from extensions.ext_redis import redis_client
from extensions.ext_storage import storage
from libs import gmpy2_pkcs10aep_cipher

# Tenant private keys are the root secret protecting every credential
# `encrypt`/`decrypt` in this module ever handles (e.g. provider API keys) --
# a raw storage-backend compromise (misconfigured bucket ACL, backup leak,
# etc.) must not hand over usable keys on its own. Private keys are therefore
# encrypted at rest with a key derived from the server's own SECRET_KEY via
# HKDF, using AES-256-GCM (authenticated, so a tampered/corrupted blob is
# detected rather than silently mis-decrypted).
_PRIVATE_KEY_AT_REST_PREFIX = b"ENCPK1:"
_PRIVATE_KEY_HKDF_CONTEXT = b"dify-tenant-privkey-at-rest-v1"


def _derive_private_key_encryption_key() -> bytes:
    """Derive the AES-256 key protecting private keys at rest from SECRET_KEY.

    Re-derived on every call rather than cached/persisted -- SECRET_KEY is
    the single source of truth, and HKDF is cheap enough that there is no
    reason to keep a copy of the derived key alive any longer than needed.
    """
    derived = HKDF(
        master=dify_config.SECRET_KEY.encode(),
        key_len=32,
        salt=b"",  # HKDF treats an empty salt as a zero-filled string of the hash's digest size
        hashmod=SHA256,
        context=_PRIVATE_KEY_HKDF_CONTEXT,
    )
    # HKDF's return type covers the num_keys > 1 case (a tuple); with the
    # default num_keys=1 it always returns a single `bytes`.
    assert isinstance(derived, bytes)
    return derived


def _encrypt_private_key_at_rest(pem_private: bytes) -> bytes:
    key = _derive_private_key_encryption_key()
    cipher = AES.new(key, AES.MODE_GCM)
    ciphertext, tag = cipher.encrypt_and_digest(pem_private)
    return _PRIVATE_KEY_AT_REST_PREFIX + cipher.nonce + tag + ciphertext


def _decrypt_private_key_at_rest(data: bytes) -> bytes:
    """Undo `_encrypt_private_key_at_rest`.

    Falls back to returning `data` unchanged when it doesn't carry the
    encrypted-at-rest prefix, so tenant private keys written before this
    protection was added keep loading exactly as before -- no forced
    migration/rewrite of existing keys.
    """
    if not data.startswith(_PRIVATE_KEY_AT_REST_PREFIX):
        return data

    payload = data[len(_PRIVATE_KEY_AT_REST_PREFIX) :]
    nonce, tag, ciphertext = payload[:16], payload[16:32], payload[32:]
    key = _derive_private_key_encryption_key()
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    return cipher.decrypt_and_verify(ciphertext, tag)


def generate_key_pair(tenant_id: str) -> str:
    private_key = RSA.generate(2048)
    public_key = private_key.publickey()

    pem_private = private_key.export_key()
    pem_public = public_key.export_key()

    filepath = f"privkeys/{tenant_id}/private.pem"

    storage.save(filepath, _encrypt_private_key_at_rest(pem_private))

    return pem_public.decode()


prefix_hybrid = b"HYBRID:"


def encrypt(text: str, public_key: Union[str, bytes]) -> bytes:
    if isinstance(public_key, str):
        public_key = public_key.encode()

    aes_key = get_random_bytes(16)
    cipher_aes = AES.new(aes_key, AES.MODE_EAX)

    ciphertext, tag = cipher_aes.encrypt_and_digest(text.encode())

    rsa_key = RSA.import_key(public_key)
    cipher_rsa = gmpy2_pkcs10aep_cipher.new(rsa_key)

    enc_aes_key: bytes = cipher_rsa.encrypt(aes_key)

    encrypted_data = enc_aes_key + cipher_aes.nonce + tag + ciphertext

    return prefix_hybrid + encrypted_data


def get_decrypt_decoding(tenant_id: str) -> tuple[RSA.RsaKey, object]:
    filepath = f"privkeys/{tenant_id}/private.pem"

    cache_key = f"tenant_privkey:{hashlib.sha3_256(filepath.encode()).hexdigest()}"
    private_key_at_rest = redis_client.get(cache_key)
    if not private_key_at_rest:
        try:
            private_key_at_rest = storage.load(filepath)
        except FileNotFoundError:
            raise PrivkeyNotFoundError(f"Private key not found, tenant_id: {tenant_id}")

        # Cache the at-rest (possibly encrypted) form, not the decrypted PEM,
        # so the plaintext key material lives no longer than each individual
        # decrypt/import call needs it for.
        redis_client.setex(cache_key, 120, private_key_at_rest)

    private_key = _decrypt_private_key_at_rest(private_key_at_rest)

    rsa_key = RSA.import_key(private_key)
    cipher_rsa = gmpy2_pkcs10aep_cipher.new(rsa_key)

    return rsa_key, cipher_rsa


def decrypt_token_with_decoding(encrypted_text: bytes, rsa_key: RSA.RsaKey, cipher_rsa) -> str:
    if encrypted_text.startswith(prefix_hybrid):
        encrypted_text = encrypted_text[len(prefix_hybrid) :]

        enc_aes_key = encrypted_text[: rsa_key.size_in_bytes()]
        nonce = encrypted_text[rsa_key.size_in_bytes() : rsa_key.size_in_bytes() + 16]
        tag = encrypted_text[rsa_key.size_in_bytes() + 16 : rsa_key.size_in_bytes() + 32]
        ciphertext = encrypted_text[rsa_key.size_in_bytes() + 32 :]

        aes_key = cipher_rsa.decrypt(enc_aes_key)

        cipher_aes = AES.new(aes_key, AES.MODE_EAX, nonce=nonce)
        decrypted_text = cipher_aes.decrypt_and_verify(ciphertext, tag)
    else:
        decrypted_text = cipher_rsa.decrypt(encrypted_text)

    return decrypted_text.decode()


def decrypt(encrypted_text: bytes, tenant_id: str) -> str:
    rsa_key, cipher_rsa = get_decrypt_decoding(tenant_id)

    return decrypt_token_with_decoding(encrypted_text=encrypted_text, rsa_key=rsa_key, cipher_rsa=cipher_rsa)


class PrivkeyNotFoundError(Exception):
    pass
