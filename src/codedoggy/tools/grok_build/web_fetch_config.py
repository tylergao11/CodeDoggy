"""Runtime-configurable parameters for web_fetch.

Ported from:
  grok-build/.../implementations/grok_build/web_fetch/config.rs
    MAX_URL_LENGTH, MAX_REDIRECTS, USER_AGENT_STRING
    WebFetchParams + defaults
    DEFAULT_ALLOWED_DOMAINS
"""

from __future__ import annotations

from dataclasses import dataclass

# Safety-boundary constants. Not configurable. (config.rs)
MAX_URL_LENGTH: int = 2_000
MAX_REDIRECTS: int = 10
USER_AGENT_STRING: str = "Mozilla/5.0 (compatible; grok-agent/1.0; +https://x.ai)"

ACCEPT_HEADER: str = (
    "text/markdown,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
)
ACCEPT_LANGUAGE: str = "en-US,en;q=0.9"

# Defaults (WebFetchParams::*_or methods)
DEFAULT_CACHE_TTL_SECS: int = 15 * 60
DEFAULT_MAX_CACHE_ENTRIES: int = 128
DEFAULT_TIMEOUT_SECS: float = 60.0
DEFAULT_MAX_CONTENT_LENGTH: int = 10 * 1024 * 1024
DEFAULT_MAX_MARKDOWN_LENGTH: int = 100_000
DEFAULT_CONTEXT_WINDOW_TOKENS: int = 128_000

# overflow.rs
WEB_FETCH_CONTEXT_PERCENT: float = 0.03
BYTES_PER_TOKEN: int = 4  # xai_token_estimation::BYTES_PER_TOKEN


@dataclass
class WebFetchParams:
    """Runtime-configurable parameters (Grok ``WebFetchParams``)."""

    cache_ttl_secs: int | None = None
    max_cache_entries: int | None = None
    timeout_secs: float | None = None
    max_content_length: int | None = None
    max_markdown_length: int | None = None
    context_window_tokens: int | None = None
    # None = do not enforce allowlist in the client layer (Grok enforces via
    # permission manager). Empty list = block all. Non-empty = DomainMatcher.
    allowed_domains: list[str] | None = None
    proxy_endpoint: str | None = None

    def get_cache_ttl_secs(self) -> float:
        return float(self.cache_ttl_secs if self.cache_ttl_secs is not None else DEFAULT_CACHE_TTL_SECS)

    def get_max_cache_entries(self) -> int:
        return self.max_cache_entries if self.max_cache_entries is not None else DEFAULT_MAX_CACHE_ENTRIES

    def get_timeout_secs(self) -> float:
        return float(self.timeout_secs if self.timeout_secs is not None else DEFAULT_TIMEOUT_SECS)

    def get_max_content_length(self) -> int:
        return (
            self.max_content_length
            if self.max_content_length is not None
            else DEFAULT_MAX_CONTENT_LENGTH
        )

    def get_max_markdown_length(self) -> int:
        return (
            self.max_markdown_length
            if self.max_markdown_length is not None
            else DEFAULT_MAX_MARKDOWN_LENGTH
        )

    def get_context_window_tokens(self) -> int:
        return (
            self.context_window_tokens
            if self.context_window_tokens is not None
            else DEFAULT_CONTEXT_WINDOW_TOKENS
        )

    def get_allowed_domains_default_list(self) -> list[str]:
        """Grok ``WebFetchParams::allowed_domains`` — None → DEFAULT_ALLOWED_DOMAINS."""
        if self.allowed_domains is not None:
            return list(self.allowed_domains)
        return list(DEFAULT_ALLOWED_DOMAINS)


# Default allowlist for web_fetch tool (config.rs DEFAULT_ALLOWED_DOMAINS).
# Note: GET-only preapproved domains. Path-scoped entries included as-is.
DEFAULT_ALLOWED_DOMAINS: list[str] = [
    # xAI
    "x.ai",
    "console.x.ai",
    "docs.x.ai",
    "api.x.ai",
    # Programming languages
    "docs.python.org",
    "en.cppreference.com",
    "docs.oracle.com",
    "learn.microsoft.com",
    "developer.mozilla.org",
    "go.dev",
    "pkg.go.dev",
    "www.php.net",
    "docs.swift.org",
    "kotlinlang.org",
    "ruby-doc.org",
    "doc.rust-lang.org",
    "docs.rs",
    "www.typescriptlang.org",
    # Web and JS frameworks
    "react.dev",
    "angular.io",
    "vuejs.org",
    "nextjs.org",
    "expressjs.com",
    "nodejs.org",
    "bun.sh",
    "jquery.com",
    "getbootstrap.com",
    "tailwindcss.com",
    "d3js.org",
    "threejs.org",
    "redux.js.org",
    "webpack.js.org",
    "jestjs.io",
    "reactrouter.com",
    # Python frameworks
    "docs.djangoproject.com",
    "flask.palletsprojects.com",
    "fastapi.tiangolo.com",
    "pandas.pydata.org",
    "numpy.org",
    "www.tensorflow.org",
    "pytorch.org",
    "scikit-learn.org",
    "matplotlib.org",
    "requests.readthedocs.io",
    "jupyter.org",
    # PHP frameworks
    "laravel.com",
    "symfony.com",
    "wordpress.org",
    # Java frameworks
    "docs.spring.io",
    "hibernate.org",
    "tomcat.apache.org",
    "gradle.org",
    "maven.apache.org",
    # .NET
    "asp.net",
    "dotnet.microsoft.com",
    "nuget.org",
    "blazor.net",
    # Mobile
    "reactnative.dev",
    "docs.flutter.dev",
    "developer.apple.com",
    "developer.android.com",
    # Data science / ML
    "keras.io",
    "spark.apache.org",
    "huggingface.co",
    "www.kaggle.com",
    # Databases
    "redis.io",
    "www.postgresql.org",
    "dev.mysql.com",
    "www.sqlite.org",
    "graphql.org",
    "prisma.io",
    # Cloud and DevOps
    "docs.aws.amazon.com",
    "cloud.google.com",
    "kubernetes.io",
    "www.docker.com",
    "www.terraform.io",
    "www.ansible.com",
    "vercel.com/docs",
    "docs.netlify.com",
    "devcenter.heroku.com",
    # Testing and monitoring
    "cypress.io",
    "selenium.dev",
    # Game development
    "docs.unity.com",
    "docs.unrealengine.com",
    # Other tools
    "git-scm.com",
    "nginx.org",
    "httpd.apache.org",
]
