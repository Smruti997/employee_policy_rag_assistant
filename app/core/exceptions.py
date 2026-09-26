"""Application-specific exceptions."""


class AuthenticationError(Exception):
    """Raised when a client cannot establish an authenticated session."""


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is absent or unsafe."""


class OcrUnavailableError(RuntimeError):
    """Raised when a scanned page needs OCR but the Tesseract binary is missing."""


class ChatServiceError(RuntimeError):
    """Safe, user-facing failure raised while processing an authenticated turn."""
