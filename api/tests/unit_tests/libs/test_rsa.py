from unittest.mock import MagicMock, patch

import pytest
from Crypto.PublicKey import RSA

from libs import gmpy2_pkcs10aep_cipher, rsa


def test_gmpy2_pkcs10aep_cipher():
    rsa_key = RSA.generate(2048)
    public_key = rsa_key.publickey().export_key(format="PEM")
    private_key = rsa_key.export_key(format="PEM")

    public_rsa_key = RSA.import_key(public_key)
    public_cipher_rsa2 = gmpy2_pkcs10aep_cipher.new(public_rsa_key)

    private_rsa_key = RSA.import_key(private_key)
    private_cipher_rsa = gmpy2_pkcs10aep_cipher.new(private_rsa_key)

    raw_text = "raw_text"
    raw_text_bytes = raw_text.encode()

    # RSA encryption by public key and decryption by private key
    encrypted_by_pub_key = public_cipher_rsa2.encrypt(message=raw_text_bytes)
    decrypted_by_pub_key = private_cipher_rsa.decrypt(encrypted_by_pub_key)
    assert decrypted_by_pub_key == raw_text_bytes

    # RSA encryption and decryption by private key
    encrypted_by_private_key = private_cipher_rsa.encrypt(message=raw_text_bytes)
    decrypted_by_private_key = private_cipher_rsa.decrypt(encrypted_by_private_key)
    assert decrypted_by_private_key == raw_text_bytes


class TestPrivateKeyAtRestEncryption:
    """Regression coverage for the confirmed Medium-severity finding: tenant
    private keys were written to storage as plaintext PEM, so a raw
    storage-backend compromise alone (misconfigured bucket ACL, backup leak,
    etc.) would hand over every credential `encrypt`/`decrypt` in this module
    ever protects."""

    def _pem_private_key(self) -> bytes:
        return RSA.generate(2048).export_key()

    def test_round_trips(self):
        pem_private = self._pem_private_key()

        with patch("libs.rsa.dify_config.SECRET_KEY", "test-secret-key"):
            at_rest = rsa._encrypt_private_key_at_rest(pem_private)
            recovered = rsa._decrypt_private_key_at_rest(at_rest)

        assert recovered == pem_private

    def test_encrypted_form_does_not_contain_the_plaintext_pem(self):
        pem_private = self._pem_private_key()

        with patch("libs.rsa.dify_config.SECRET_KEY", "test-secret-key"):
            at_rest = rsa._encrypt_private_key_at_rest(pem_private)

        assert pem_private not in at_rest
        assert at_rest.startswith(rsa._PRIVATE_KEY_AT_REST_PREFIX)

    def test_legacy_plaintext_pem_passes_through_unchanged(self):
        """A private key written before this protection was added has no
        encrypted-at-rest prefix -- it must keep loading exactly as before,
        with no forced migration/rewrite."""
        pem_private = self._pem_private_key()

        with patch("libs.rsa.dify_config.SECRET_KEY", "test-secret-key"):
            recovered = rsa._decrypt_private_key_at_rest(pem_private)

        assert recovered == pem_private

    def test_tampered_ciphertext_is_rejected(self):
        pem_private = self._pem_private_key()

        with patch("libs.rsa.dify_config.SECRET_KEY", "test-secret-key"):
            at_rest = rsa._encrypt_private_key_at_rest(pem_private)
            tampered = at_rest[:-1] + bytes([at_rest[-1] ^ 0xFF])

            with pytest.raises(ValueError):
                rsa._decrypt_private_key_at_rest(tampered)

    def test_decryption_fails_under_a_different_secret_key(self):
        pem_private = self._pem_private_key()

        with patch("libs.rsa.dify_config.SECRET_KEY", "test-secret-key-one"):
            at_rest = rsa._encrypt_private_key_at_rest(pem_private)

        with (
            patch("libs.rsa.dify_config.SECRET_KEY", "test-secret-key-two"),
            pytest.raises(ValueError),
        ):
            rsa._decrypt_private_key_at_rest(at_rest)


class TestGenerateKeyPairStoresEncryptedPrivateKey:
    """End-to-end (storage + redis mocked) coverage of `generate_key_pair` /
    `get_decrypt_decoding`, proving the full pipeline round-trips and that
    what actually lands in storage is not the raw PEM."""

    def test_generated_private_key_is_encrypted_at_rest_and_round_trips(self):
        tenant_id = "tenant-generated"
        saved: dict[str, bytes] = {}

        mock_storage = MagicMock()
        mock_storage.save.side_effect = lambda path, data: saved.__setitem__(path, data)
        mock_storage.load.side_effect = lambda path: saved[path]

        mock_redis = MagicMock()
        mock_redis.get.return_value = None  # force a storage load, not a cache hit

        with (
            patch("libs.rsa.dify_config.SECRET_KEY", "test-secret-key"),
            patch("libs.rsa.storage", mock_storage),
            patch("libs.rsa.redis_client", mock_redis),
        ):
            public_key_pem = rsa.generate_key_pair(tenant_id)

            filepath = f"privkeys/{tenant_id}/private.pem"
            stored_bytes = saved[filepath]
            assert stored_bytes.startswith(rsa._PRIVATE_KEY_AT_REST_PREFIX)
            assert b"-----BEGIN" not in stored_bytes

            plaintext = "a provider api key"
            ciphertext = rsa.encrypt(plaintext, public_key_pem)
            decrypted = rsa.decrypt(ciphertext, tenant_id)

        assert decrypted == plaintext
        # What was cached in redis is the at-rest (encrypted) form, not a
        # decrypted copy of the plaintext key material.
        cached_value = mock_redis.setex.call_args.args[2]
        assert cached_value == stored_bytes

    def test_pre_existing_plaintext_private_key_still_works(self):
        """A tenant whose private key was written before this fix (raw PEM,
        no encryption) must keep working without any migration step."""
        tenant_id = "tenant-legacy"
        rsa_key = RSA.generate(2048)
        legacy_pem_private = rsa_key.export_key()
        legacy_pem_public = rsa_key.publickey().export_key()

        saved = {f"privkeys/{tenant_id}/private.pem": legacy_pem_private}

        mock_storage = MagicMock()
        mock_storage.load.side_effect = lambda path: saved[path]

        mock_redis = MagicMock()
        mock_redis.get.return_value = None

        with (
            patch("libs.rsa.dify_config.SECRET_KEY", "test-secret-key"),
            patch("libs.rsa.storage", mock_storage),
            patch("libs.rsa.redis_client", mock_redis),
        ):
            plaintext = "a provider api key"
            ciphertext = rsa.encrypt(plaintext, legacy_pem_public)
            decrypted = rsa.decrypt(ciphertext, tenant_id)

        assert decrypted == plaintext
