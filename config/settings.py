import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured

BASE_DIR = Path(__file__).resolve().parent.parent
ON_VERCEL = os.environ.get("VERCEL") == "1"
IS_VERCEL_BUILD = ON_VERCEL and os.environ.get("CI") == "1"
DEBUG = False if ON_VERCEL else os.environ.get("DJANGO_DEBUG", "1") == "1"

try:
    from . import local_settings
except ImportError:
    local_settings = None


def configured(name, default=""):
    return os.environ.get(name, getattr(local_settings, name, default))


SECRET_KEY = configured("DJANGO_SECRET_KEY")
EULERPOOL_API_KEY = configured("EULERPOOL_API_KEY")
if not SECRET_KEY:
    if IS_VERCEL_BUILD:
        SECRET_KEY = "vercel-build-only-7f3c9e1a5b8d2f6c4a0e9b7d3f1c8a5e"
    else:
        raise ImproperlyConfigured("Set DJANGO_SECRET_KEY or create config/local_settings.py.")

ALLOWED_HOSTS = ["localhost", "127.0.0.1", "[::1]"]
for hostname in [
    os.environ.get("VERCEL_URL", ""),
    os.environ.get("VERCEL_PROJECT_PRODUCTION_URL", ""),
    *os.environ.get("DJANGO_ALLOWED_HOSTS", "").split(","),
]:
    if hostname.strip():
        ALLOWED_HOSTS.append(hostname.strip())

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.humanize",
    "django.contrib.staticfiles",
    "dashboard",
]
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]
ROOT_URLCONF = "config.urls"
WSGI_APPLICATION = "config.wsgi.application"
TEMPLATES = [{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [],
    "APP_DIRS": True,
    "OPTIONS": {"context_processors": [
        "django.template.context_processors.request",
        "django.contrib.auth.context_processors.auth",
        "django.contrib.messages.context_processors.messages",
    ]},
}]

DATABASES = {"default": {
    "ENGINE": "mysql.connector.django",
    "NAME": configured("MYSQL_DATABASE"),
    "USER": configured("MYSQL_USER"),
    "PASSWORD": configured("MYSQL_PASSWORD"),
    "HOST": configured("MYSQL_HOST"),
    "PORT": configured("MYSQL_PORT", "3306"),
    "CONN_MAX_AGE": 0,
    "OPTIONS": {
        "charset": "utf8mb4",
        "use_pure": True,
        "connection_timeout": 10,
        "init_command": "SET sql_mode='STRICT_TRANS_TABLES'",
        "isolation_level": "read committed",
    },
}}
AUTH_PASSWORD_VALIDATORS = []
LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "dashboard"
LOGOUT_REDIRECT_URL = "login"
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SECURE = ON_VERCEL
SESSION_COOKIE_SAMESITE = "Lax"
SESSION_COOKIE_AGE = 604800
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_TZ = True
STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

DATA_UPLOAD_MAX_MEMORY_SIZE = 2_000_000
FILE_UPLOAD_MAX_MEMORY_SIZE = 2_000_000
FILE_UPLOAD_HANDLERS = ["django.core.files.uploadhandler.MemoryFileUploadHandler"]
SECURE_CONTENT_TYPE_NOSNIFF = True
CSRF_COOKIE_HTTPONLY = True
CSRF_COOKIE_SECURE = ON_VERCEL
SECURE_SSL_REDIRECT = ON_VERCEL
SECURE_HSTS_SECONDS = 31_536_000 if ON_VERCEL else 0
SECURE_HSTS_INCLUDE_SUBDOMAINS = ON_VERCEL
SECURE_HSTS_PRELOAD = ON_VERCEL
if ON_VERCEL:
    SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
