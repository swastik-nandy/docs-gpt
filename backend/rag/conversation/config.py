from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import rtoml
from psycopg.conninfo import make_conninfo


DEFAULT_CONVERSATION_CONFIG_PATH = Path(__file__).with_name("config.toml")


@dataclass(frozen=True, slots=True)
class ConversationDatabaseBehaviorConfig:
    enabled: bool
    required: bool
    connect_timeout: int
    application_name: str

    def validate(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError(
                "database.conversation.enabled must be boolean."
            )

        if type(self.required) is not bool:
            raise ValueError(
                "database.conversation.required must be boolean."
            )

        if self.connect_timeout <= 0:
            raise ValueError(
                "database.conversation.connect_timeout "
                "must be greater than zero."
            )

        if not self.application_name.strip():
            raise ValueError(
                "database.conversation.application_name cannot be empty."
            )


@dataclass(frozen=True, slots=True)
class ConversationPoolConfig:
    min_size: int
    max_size: int
    acquire_timeout_seconds: float
    startup_timeout_seconds: float
    shutdown_timeout_seconds: float
    max_waiting: int
    max_lifetime_seconds: float
    max_idle_seconds: float
    reconnect_timeout_seconds: float
    worker_count: int
    check_connection_on_acquire: bool

    def validate(self) -> None:
        if self.min_size < 0:
            raise ValueError(
                "database.conversation.pool.min_size cannot be negative."
            )

        if self.max_size <= 0:
            raise ValueError(
                "database.conversation.pool.max_size "
                "must be greater than zero."
            )

        if self.min_size > self.max_size:
            raise ValueError(
                "database.conversation.pool.min_size cannot be greater "
                "than database.conversation.pool.max_size."
            )

        positive_numbers = {
            "acquire_timeout_seconds": self.acquire_timeout_seconds,
            "startup_timeout_seconds": self.startup_timeout_seconds,
            "shutdown_timeout_seconds": self.shutdown_timeout_seconds,
            "max_lifetime_seconds": self.max_lifetime_seconds,
            "reconnect_timeout_seconds": self.reconnect_timeout_seconds,
        }

        for field_name, value in positive_numbers.items():
            if value <= 0:
                raise ValueError(
                    f"database.conversation.pool.{field_name} "
                    "must be greater than zero."
                )

        if self.max_idle_seconds < 0:
            raise ValueError(
                "database.conversation.pool.max_idle_seconds "
                "cannot be negative."
            )

        if self.max_waiting < 0:
            raise ValueError(
                "database.conversation.pool.max_waiting "
                "cannot be negative."
            )

        if self.worker_count <= 0:
            raise ValueError(
                "database.conversation.pool.worker_count "
                "must be greater than zero."
            )

        if type(self.check_connection_on_acquire) is not bool:
            raise ValueError(
                "database.conversation.pool."
                "check_connection_on_acquire must be boolean."
            )


@dataclass(frozen=True, slots=True)
class StatefulRetrievalConfig:
    enabled: bool
    turn_window_limit: int
    entity_window_limit: int

    def validate(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError(
                "stateful_retrieval.enabled must be boolean."
            )

        if self.turn_window_limit <= 0:
            raise ValueError(
                "stateful_retrieval.turn_window_limit "
                "must be greater than zero."
            )

        if self.entity_window_limit <= 0:
            raise ValueError(
                "stateful_retrieval.entity_window_limit "
                "must be greater than zero."
            )


@dataclass(frozen=True, slots=True)
class ReferenceResolutionConfig:
    enabled: bool
    confidence_for_recent_entity_substitution: float
    recent_turn_fallback_confidence_multiplier: float
    max_recent_entities: int
    max_recent_turns_for_fallback: int

    def validate(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError(
                "reference_resolution.enabled must be boolean."
            )

        _validate_probability(
            name=(
                "reference_resolution."
                "confidence_for_recent_entity_substitution"
            ),
            value=self.confidence_for_recent_entity_substitution,
        )

        _validate_probability(
            name=(
                "reference_resolution."
                "recent_turn_fallback_confidence_multiplier"
            ),
            value=self.recent_turn_fallback_confidence_multiplier,
        )

        if self.max_recent_entities <= 0:
            raise ValueError(
                "reference_resolution.max_recent_entities "
                "must be greater than zero."
            )

        if self.max_recent_turns_for_fallback <= 0:
            raise ValueError(
                "reference_resolution.max_recent_turns_for_fallback "
                "must be greater than zero."
            )


@dataclass(frozen=True, slots=True)
class EntityExtractionConfidenceConfig:
    code_or_exact: float
    quoted_phrase: float
    identifier: float
    acronym: float
    proper_noun: float
    unknown: float

    def validate(self) -> None:
        for entity_type, confidence in self.as_mapping().items():
            _validate_probability(
                name=(
                    "entity_extraction.confidence."
                    f"{entity_type}"
                ),
                value=confidence,
            )

    def as_mapping(self) -> dict[str, float]:
        return {
            "code_or_exact": self.code_or_exact,
            "quoted_phrase": self.quoted_phrase,
            "identifier": self.identifier,
            "acronym": self.acronym,
            "proper_noun": self.proper_noun,
            "unknown": self.unknown,
        }

    def for_entity_type(
        self,
        entity_type: str,
    ) -> float:
        normalized_type = (
            entity_type or ""
        ).strip().casefold()

        return self.as_mapping().get(
            normalized_type,
            self.unknown,
        )


@dataclass(frozen=True, slots=True)
class EntityExtractionConfig:
    enabled: bool

    max_entities: int
    minimum_entity_length: int
    max_entity_length: int
    max_proper_noun_tokens: int
    max_acronym_length: int

    extract_backtick_phrases: bool
    extract_quoted_phrases: bool
    extract_qualified_identifiers: bool
    extract_acronyms: bool
    extract_proper_nouns: bool

    strip_leading_function_words: bool
    reject_sentence_initial_single_token: bool

    leading_function_words: tuple[str, ...]
    standalone_non_entities: tuple[str, ...]

    confidence: EntityExtractionConfidenceConfig

    def validate(self) -> None:
        boolean_fields = {
            "enabled": self.enabled,
            "extract_backtick_phrases": (
                self.extract_backtick_phrases
            ),
            "extract_quoted_phrases": (
                self.extract_quoted_phrases
            ),
            "extract_qualified_identifiers": (
                self.extract_qualified_identifiers
            ),
            "extract_acronyms": self.extract_acronyms,
            "extract_proper_nouns": (
                self.extract_proper_nouns
            ),
            "strip_leading_function_words": (
                self.strip_leading_function_words
            ),
            "reject_sentence_initial_single_token": (
                self.reject_sentence_initial_single_token
            ),
        }

        for field_name, value in boolean_fields.items():
            if type(value) is not bool:
                raise ValueError(
                    f"entity_extraction.{field_name} "
                    "must be boolean."
                )

        positive_integer_fields = {
            "max_entities": self.max_entities,
            "minimum_entity_length": (
                self.minimum_entity_length
            ),
            "max_entity_length": self.max_entity_length,
            "max_proper_noun_tokens": (
                self.max_proper_noun_tokens
            ),
            "max_acronym_length": (
                self.max_acronym_length
            ),
        }

        for field_name, value in positive_integer_fields.items():
            if value <= 0:
                raise ValueError(
                    f"entity_extraction.{field_name} "
                    "must be greater than zero."
                )

        if self.minimum_entity_length > self.max_entity_length:
            raise ValueError(
                "entity_extraction.minimum_entity_length cannot be "
                "greater than entity_extraction.max_entity_length."
            )

        if self.max_acronym_length > self.max_entity_length:
            raise ValueError(
                "entity_extraction.max_acronym_length cannot be "
                "greater than entity_extraction.max_entity_length."
            )

        _validate_unique_non_empty_strings(
            name=(
                "entity_extraction."
                "leading_function_words"
            ),
            values=self.leading_function_words,
        )

        _validate_unique_non_empty_strings(
            name=(
                "entity_extraction."
                "standalone_non_entities"
            ),
            values=self.standalone_non_entities,
        )

        self.confidence.validate()


@dataclass(frozen=True, slots=True)
class ConversationRuntimeConfig:
    database: ConversationDatabaseBehaviorConfig
    pool: ConversationPoolConfig
    stateful_retrieval: StatefulRetrievalConfig
    reference_resolution: ReferenceResolutionConfig
    entity_extraction: EntityExtractionConfig

    def validate(self) -> None:
        self.database.validate()
        self.pool.validate()
        self.stateful_retrieval.validate()
        self.reference_resolution.validate()
        self.entity_extraction.validate()


@dataclass(frozen=True, slots=True)
class ConversationDbConfig:
    dsn: str
    behavior: ConversationDatabaseBehaviorConfig
    pool: ConversationPoolConfig

    @property
    def enabled(self) -> bool:
        return self.behavior.enabled

    @property
    def required(self) -> bool:
        return self.behavior.required

    @property
    def connect_timeout(self) -> int:
        return self.behavior.connect_timeout

    @classmethod
    def from_env(
        cls,
        *,
        runtime_config: ConversationRuntimeConfig,
    ) -> ConversationDbConfig:
        runtime_config.validate()
        load_env_file()

        behavior = runtime_config.database

        if not behavior.enabled:
            return cls(
                dsn="",
                behavior=behavior,
                pool=runtime_config.pool,
            )

        explicit_dsn = _env_clean(
            "CONVERSATION_DATABASE_URL"
        )

        if explicit_dsn:
            return cls(
                dsn=make_conninfo(
                    explicit_dsn,
                    connect_timeout=behavior.connect_timeout,
                    application_name=behavior.application_name,
                ),
                behavior=behavior,
                pool=runtime_config.pool,
            )

        database_name = _env_clean(
            "CONVERSATION_DB_NAME"
        )

        if not database_name:
            if behavior.required:
                raise ValueError(
                    "CONVERSATION_DB_NAME is required when "
                    "CONVERSATION_DATABASE_URL is not set."
                )

            return cls(
                dsn="",
                behavior=behavior,
                pool=runtime_config.pool,
            )

        connection_parameters: dict[str, str] = {
            "dbname": database_name,
            "connect_timeout": str(
                behavior.connect_timeout
            ),
            "application_name": (
                behavior.application_name
            ),
        }

        optional_environment_fields = {
            "CONVERSATION_DB_HOST": "host",
            "CONVERSATION_DB_PORT": "port",
            "CONVERSATION_DB_USER": "user",
            "CONVERSATION_DB_PASSWORD": "password",
        }

        for environment_name, connection_key in (
            optional_environment_fields.items()
        ):
            value = _env_clean(environment_name)
            if value:
                connection_parameters[connection_key] = value

        return cls(
            dsn=make_conninfo(
                **connection_parameters
            ),
            behavior=behavior,
            pool=runtime_config.pool,
        )

    def validate(self) -> None:
        self.behavior.validate()
        self.pool.validate()

        if (
            self.behavior.enabled
            and self.behavior.required
            and not self.dsn.strip()
        ):
            raise ValueError(
                "conversation database DSN cannot be empty "
                "when persistence is enabled and required."
            )


def load_conversation_runtime_config(
    path: str | Path | None = None,
) -> ConversationRuntimeConfig:
    config_path = (
        Path(path).expanduser().resolve()
        if path is not None
        else _conversation_config_path()
    )

    try:
        raw_text = config_path.read_text(
            encoding="utf-8"
        )
    except OSError as exc:
        raise RuntimeError(
            "Could not read conversation config: "
            f"{config_path}"
        ) from exc

    try:
        raw_config = rtoml.loads(raw_text)
    except Exception as exc:
        raise RuntimeError(
            "Could not parse conversation TOML config: "
            f"{config_path}"
        ) from exc

    config = parse_conversation_runtime_config(
        raw_config
    )
    config.validate()

    return config


def parse_conversation_runtime_config(
    raw_config: Mapping[str, Any],
) -> ConversationRuntimeConfig:
    database_section = _required_section(
        raw_config,
        "database",
    )

    conversation_database_section = _required_section(
        database_section,
        "conversation",
        path="database",
    )

    pool_section = _required_section(
        conversation_database_section,
        "pool",
        path="database.conversation",
    )

    stateful_section = _required_section(
        raw_config,
        "stateful_retrieval",
    )

    reference_section = _required_section(
        raw_config,
        "reference_resolution",
    )

    entity_section = _required_section(
        raw_config,
        "entity_extraction",
    )

    confidence_section = _required_section(
        entity_section,
        "confidence",
        path="entity_extraction",
    )

    config = ConversationRuntimeConfig(
        database=ConversationDatabaseBehaviorConfig(
            enabled=_required_bool(
                conversation_database_section,
                "enabled",
                path="database.conversation",
            ),
            required=_required_bool(
                conversation_database_section,
                "required",
                path="database.conversation",
            ),
            connect_timeout=_required_int(
                conversation_database_section,
                "connect_timeout",
                path="database.conversation",
            ),
            application_name=_required_string(
                conversation_database_section,
                "application_name",
                path="database.conversation",
            ),
        ),
        pool=ConversationPoolConfig(
            min_size=_required_int(
                pool_section,
                "min_size",
                path="database.conversation.pool",
            ),
            max_size=_required_int(
                pool_section,
                "max_size",
                path="database.conversation.pool",
            ),
            acquire_timeout_seconds=_required_float(
                pool_section,
                "acquire_timeout_seconds",
                path="database.conversation.pool",
            ),
            startup_timeout_seconds=_required_float(
                pool_section,
                "startup_timeout_seconds",
                path="database.conversation.pool",
            ),
            shutdown_timeout_seconds=_required_float(
                pool_section,
                "shutdown_timeout_seconds",
                path="database.conversation.pool",
            ),
            max_waiting=_required_int(
                pool_section,
                "max_waiting",
                path="database.conversation.pool",
            ),
            max_lifetime_seconds=_required_float(
                pool_section,
                "max_lifetime_seconds",
                path="database.conversation.pool",
            ),
            max_idle_seconds=_required_float(
                pool_section,
                "max_idle_seconds",
                path="database.conversation.pool",
            ),
            reconnect_timeout_seconds=_required_float(
                pool_section,
                "reconnect_timeout_seconds",
                path="database.conversation.pool",
            ),
            worker_count=_required_int(
                pool_section,
                "worker_count",
                path="database.conversation.pool",
            ),
            check_connection_on_acquire=_required_bool(
                pool_section,
                "check_connection_on_acquire",
                path="database.conversation.pool",
            ),
        ),
        stateful_retrieval=StatefulRetrievalConfig(
            enabled=_required_bool(
                stateful_section,
                "enabled",
                path="stateful_retrieval",
            ),
            turn_window_limit=_required_int(
                stateful_section,
                "turn_window_limit",
                path="stateful_retrieval",
            ),
            entity_window_limit=_required_int(
                stateful_section,
                "entity_window_limit",
                path="stateful_retrieval",
            ),
        ),
        reference_resolution=ReferenceResolutionConfig(
            enabled=_required_bool(
                reference_section,
                "enabled",
                path="reference_resolution",
            ),
            confidence_for_recent_entity_substitution=(
                _required_float(
                    reference_section,
                    (
                        "confidence_for_recent_"
                        "entity_substitution"
                    ),
                    path="reference_resolution",
                )
            ),
            recent_turn_fallback_confidence_multiplier=(
                _required_float(
                    reference_section,
                    (
                        "recent_turn_fallback_"
                        "confidence_multiplier"
                    ),
                    path="reference_resolution",
                )
            ),
            max_recent_entities=_required_int(
                reference_section,
                "max_recent_entities",
                path="reference_resolution",
            ),
            max_recent_turns_for_fallback=_required_int(
                reference_section,
                "max_recent_turns_for_fallback",
                path="reference_resolution",
            ),
        ),
        entity_extraction=EntityExtractionConfig(
            enabled=_required_bool(
                entity_section,
                "enabled",
                path="entity_extraction",
            ),
            max_entities=_required_int(
                entity_section,
                "max_entities",
                path="entity_extraction",
            ),
            minimum_entity_length=_required_int(
                entity_section,
                "minimum_entity_length",
                path="entity_extraction",
            ),
            max_entity_length=_required_int(
                entity_section,
                "max_entity_length",
                path="entity_extraction",
            ),
            max_proper_noun_tokens=_required_int(
                entity_section,
                "max_proper_noun_tokens",
                path="entity_extraction",
            ),
            max_acronym_length=_required_int(
                entity_section,
                "max_acronym_length",
                path="entity_extraction",
            ),
            extract_backtick_phrases=_required_bool(
                entity_section,
                "extract_backtick_phrases",
                path="entity_extraction",
            ),
            extract_quoted_phrases=_required_bool(
                entity_section,
                "extract_quoted_phrases",
                path="entity_extraction",
            ),
            extract_qualified_identifiers=_required_bool(
                entity_section,
                "extract_qualified_identifiers",
                path="entity_extraction",
            ),
            extract_acronyms=_required_bool(
                entity_section,
                "extract_acronyms",
                path="entity_extraction",
            ),
            extract_proper_nouns=_required_bool(
                entity_section,
                "extract_proper_nouns",
                path="entity_extraction",
            ),
            strip_leading_function_words=_required_bool(
                entity_section,
                "strip_leading_function_words",
                path="entity_extraction",
            ),
            reject_sentence_initial_single_token=_required_bool(
                entity_section,
                (
                    "reject_sentence_initial_"
                    "single_token"
                ),
                path="entity_extraction",
            ),
            leading_function_words=_required_string_tuple(
                entity_section,
                "leading_function_words",
                path="entity_extraction",
            ),
            standalone_non_entities=(
                _required_string_tuple(
                    entity_section,
                    "standalone_non_entities",
                    path="entity_extraction",
                )
            ),
            confidence=EntityExtractionConfidenceConfig(
                code_or_exact=_required_float(
                    confidence_section,
                    "code_or_exact",
                    path=(
                        "entity_extraction."
                        "confidence"
                    ),
                ),
                quoted_phrase=_required_float(
                    confidence_section,
                    "quoted_phrase",
                    path=(
                        "entity_extraction."
                        "confidence"
                    ),
                ),
                identifier=_required_float(
                    confidence_section,
                    "identifier",
                    path=(
                        "entity_extraction."
                        "confidence"
                    ),
                ),
                acronym=_required_float(
                    confidence_section,
                    "acronym",
                    path=(
                        "entity_extraction."
                        "confidence"
                    ),
                ),
                proper_noun=_required_float(
                    confidence_section,
                    "proper_noun",
                    path=(
                        "entity_extraction."
                        "confidence"
                    ),
                ),
                unknown=_required_float(
                    confidence_section,
                    "unknown",
                    path=(
                        "entity_extraction."
                        "confidence"
                    ),
                ),
            ),
        ),
    )

    config.validate()
    return config


def load_env_file(
    *,
    env_file: str | Path | None = None,
    override: bool = False,
) -> Path | None:
    path = _resolve_env_file(env_file)

    if path is None:
        return None

    values = _parse_env_file(path)

    for key, value in values.items():
        if override or key not in os.environ:
            os.environ[key] = value

    return path


def _conversation_config_path() -> Path:
    configured_path = _env_clean(
        "CONVERSATION_CONFIG_PATH"
    )

    if configured_path:
        return (
            Path(configured_path)
            .expanduser()
            .resolve()
        )

    return DEFAULT_CONVERSATION_CONFIG_PATH.resolve()


def _required_section(
    mapping: Mapping[str, Any],
    key: str,
    *,
    path: str = "",
) -> Mapping[str, Any]:
    full_path = (
        f"{path}.{key}"
        if path
        else key
    )

    if key not in mapping:
        raise ValueError(
            f"Missing required TOML table: {full_path}."
        )

    value = mapping[key]

    if not isinstance(value, Mapping):
        raise ValueError(
            f"{full_path} must be a TOML table."
        )

    return value


def _required_value(
    mapping: Mapping[str, Any],
    key: str,
    *,
    path: str,
) -> Any:
    if key not in mapping:
        raise ValueError(
            "Missing required TOML value: "
            f"{path}.{key}."
        )

    return mapping[key]


def _required_bool(
    mapping: Mapping[str, Any],
    key: str,
    *,
    path: str,
) -> bool:
    value = _required_value(
        mapping,
        key,
        path=path,
    )

    if type(value) is not bool:
        raise ValueError(
            f"{path}.{key} must be boolean."
        )

    return value


def _required_int(
    mapping: Mapping[str, Any],
    key: str,
    *,
    path: str,
) -> int:
    value = _required_value(
        mapping,
        key,
        path=path,
    )

    if type(value) is not int:
        raise ValueError(
            f"{path}.{key} must be an integer."
        )

    return value


def _required_float(
    mapping: Mapping[str, Any],
    key: str,
    *,
    path: str,
) -> float:
    value = _required_value(
        mapping,
        key,
        path=path,
    )

    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
    ):
        raise ValueError(
            f"{path}.{key} must be numeric."
        )

    return float(value)


def _required_string(
    mapping: Mapping[str, Any],
    key: str,
    *,
    path: str,
) -> str:
    value = _required_value(
        mapping,
        key,
        path=path,
    )

    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            f"{path}.{key} must be a non-empty string."
        )

    return value.strip()


def _required_string_tuple(
    mapping: Mapping[str, Any],
    key: str,
    *,
    path: str,
) -> tuple[str, ...]:
    value = _required_value(
        mapping,
        key,
        path=path,
    )

    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
    ):
        raise ValueError(
            f"{path}.{key} must be an array of strings."
        )

    normalized_values: list[str] = []

    for index, item in enumerate(value):
        if (
            not isinstance(item, str)
            or not item.strip()
        ):
            raise ValueError(
                f"{path}.{key}[{index}] "
                "must be a non-empty string."
            )

        normalized_values.append(
            item.strip()
        )

    return tuple(normalized_values)


def _validate_probability(
    *,
    name: str,
    value: float,
) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{name} must be between 0 and 1."
        )


def _validate_unique_non_empty_strings(
    *,
    name: str,
    values: tuple[str, ...],
) -> None:
    if not values:
        raise ValueError(
            f"{name} cannot be empty."
        )

    normalized_values = [
        value.casefold()
        for value in values
    ]

    if len(normalized_values) != len(
        set(normalized_values)
    ):
        raise ValueError(
            f"{name} cannot contain duplicates."
        )


def _env_clean(
    name: str,
) -> str | None:
    value = os.getenv(name)

    if value is None:
        return None

    cleaned_value = value.strip()
    return cleaned_value or None


def _resolve_env_file(
    env_file: str | Path | None,
) -> Path | None:
    if env_file is not None:
        path = (
            Path(env_file)
            .expanduser()
            .resolve()
        )

        if not path.exists():
            raise FileNotFoundError(
                f".env file not found: {path}"
            )

        return path

    for directory in (
        Path.cwd(),
        *Path.cwd().parents,
    ):
        candidate = directory / ".env"

        if candidate.exists():
            return candidate.resolve()

    return None


def _parse_env_file(
    path: Path,
) -> dict[str, str]:
    values: dict[str, str] = {}

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    for line_number, raw_line in enumerate(
        lines,
        start=1,
    ):
        line = raw_line.strip()

        if not line or line.startswith("#"):
            continue

        if line.startswith("export "):
            line = line.removeprefix(
                "export "
            ).strip()

        if "=" not in line:
            raise ValueError(
                f"Invalid .env line {line_number} "
                f"in {path}: missing '='."
            )

        key, value = line.split(
            "=",
            1,
        )

        key = key.strip()
        value = _clean_env_value(
            value.strip()
        )

        if not key:
            raise ValueError(
                f"Invalid .env line {line_number} "
                f"in {path}: empty key."
            )

        values[key] = value

    return values


def _clean_env_value(
    value: str,
) -> str:
    if not value:
        return ""

    if (
        len(value) >= 2
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    ):
        return value[1:-1]

    comment_index = value.find(" #")

    if comment_index >= 0:
        return value[:comment_index].strip()

    return value
