"""Application-specific exceptions."""


class AuthenticationError(Exception):
    """Raised when a client cannot establish an authenticated session."""


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is absent or unsafe."""


class OcrUnavailableError(RuntimeError):
    """Raised when a scanned page needs OCR but the Tesseract binary is missing."""


class SampleDataError(RuntimeError):
    """Raised when a mock workbook is missing, empty, or missing required columns."""


class EmployeeNotFoundError(LookupError):
    """Raised when the mock data holds no record for the requested employee."""
