from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Unpack

from dotenv import load_dotenv
from loguru import logger
from pydantic import AnyHttpUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from . import constants as cs
from . import exceptions as ex
from . import logs
from .types_defs import CgrignorePatterns, ModelConfigKwargs

load_dotenv()


@dataclass
class ModelConfig:
    provider: str
    model_id: str
    api_key: str | None = None
    endpoint: str | None = None
    project_id: str | None = None
    region: str | None = None
    provider_type: str | None = None
    thinking_budget: int | None = None
    service_account_file: str | None = None

    def to_update_kwargs(self) -> ModelConfigKwargs:
        result = asdict(self)
        del result[cs.FIELD_PROVIDER]
        del result[cs.FIELD_MODEL_ID]
        return ModelConfigKwargs(**result)


class AppConfig(BaseSettings):
    """
    (H) All settings are loaded from environment variables or a .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    MEMGRAPH_HOST: str = "localhost"
    MEMGRAPH_PORT: int = 7687
    MEMGRAPH_HTTP_PORT: int = 7444
    LAB_PORT: int = 3000
    MEMGRAPH_BATCH_SIZE: int = 1000
    AGENT_RETRIES: int = 3
    ORCHESTRATOR_OUTPUT_RETRIES: int = 100

    ORCHESTRATOR_PROVIDER: str = ""
    ORCHESTRATOR_MODEL: str = ""
    ORCHESTRATOR_API_KEY: str | None = None
    ORCHESTRATOR_ENDPOINT: str | None = None
    ORCHESTRATOR_PROJECT_ID: str | None = None
    ORCHESTRATOR_REGION: str = cs.DEFAULT_REGION
    ORCHESTRATOR_PROVIDER_TYPE: str | None = None
    ORCHESTRATOR_THINKING_BUDGET: int | None = None
    ORCHESTRATOR_SERVICE_ACCOUNT_FILE: str | None = None

    CYPHER_PROVIDER: str = ""
    CYPHER_MODEL: str = ""
    CYPHER_API_KEY: str | None = None
    CYPHER_ENDPOINT: str | None = None
    CYPHER_PROJECT_ID: str | None = None
    CYPHER_REGION: str = cs.DEFAULT_REGION
    CYPHER_PROVIDER_TYPE: str | None = None
    CYPHER_THINKING_BUDGET: int | None = None
    CYPHER_SERVICE_ACCOUNT_FILE: str | None = None

    LOCAL_MODEL_ENDPOINT: AnyHttpUrl = AnyHttpUrl("http://localhost:11434/v1")

    TARGET_REPO_PATH: str = "."

    SHELL_COMMAND_TIMEOUT: int = 30
    SHELL_COMMAND_ALLOWLIST: frozenset[str] = frozenset(
        {
            "ls",
            "rg",
            "cat",
            "git",
            "echo",
            "pwd",
            "pytest",
            "mypy",
            "ruff",
            "uv",
            "find",
            "pre-commit",
            "rm",
            "cp",
            "mv",
            "mkdir",
            "rmdir",
            "wc",
            "head",
            "tail",
            "sort",
            "uniq",
            "cut",
            "tr",
            "xargs",
            "awk",
            "sed",
            "tee",
        }
    )
    SHELL_READ_ONLY_COMMANDS: frozenset[str] = frozenset(
        {
            "ls",
            "cat",
            "find",
            "pwd",
            "rg",
            "echo",
            "wc",
            "head",
            "tail",
            "sort",
            "uniq",
            "cut",
            "tr",
        }
    )
    SHELL_SAFE_GIT_SUBCOMMANDS: frozenset[str] = frozenset(
        {
            "status",
            "log",
            "diff",
            "show",
            "ls-files",
            "remote",
            "config",
            "branch",
        }
    )

    QDRANT_DB_PATH: str = "./.qdrant_code_embeddings"
    QDRANT_COLLECTION_NAME: str = "code_embeddings"
    QDRANT_VECTOR_DIM: int = 768
    QDRANT_TOP_K: int = 5
    EMBEDDING_MAX_LENGTH: int = 512
    EMBEDDING_PROGRESS_INTERVAL: int = 10

    CACHE_MAX_ENTRIES: int = 1000
    CACHE_MAX_MEMORY_MB: int = 500
    CACHE_EVICTION_DIVISOR: int = 10
    CACHE_MEMORY_THRESHOLD_RATIO: float = 0.8

    OLLAMA_HEALTH_TIMEOUT: float = 5.0

    _active_orchestrator: ModelConfig | None = None
    _active_cypher: ModelConfig | None = None

    QUIET: bool = Field(False, validation_alias="CGR_QUIET")

    def _get_default_config(self, role: str) -> ModelConfig:
        role_upper = role.upper()

        provider = getattr(self, f"{role_upper}_PROVIDER", None)
        model = getattr(self, f"{role_upper}_MODEL", None)

        if provider and model:
            return ModelConfig(
                provider=provider.lower(),
                model_id=model,
                api_key=getattr(self, f"{role_upper}_API_KEY", None),
                endpoint=getattr(self, f"{role_upper}_ENDPOINT", None),
                project_id=getattr(self, f"{role_upper}_PROJECT_ID", None),
                region=getattr(self, f"{role_upper}_REGION", cs.DEFAULT_REGION),
                provider_type=getattr(self, f"{role_upper}_PROVIDER_TYPE", None),
                thinking_budget=getattr(self, f"{role_upper}_THINKING_BUDGET", None),
                service_account_file=getattr(
                    self, f"{role_upper}_SERVICE_ACCOUNT_FILE", None
                ),
            )

        return ModelConfig(
            provider=cs.Provider.OLLAMA,
            model_id=cs.DEFAULT_MODEL,
            endpoint=str(self.LOCAL_MODEL_ENDPOINT),
            api_key=cs.DEFAULT_API_KEY,
        )

    def _get_default_orchestrator_config(self) -> ModelConfig:
        return self._get_default_config(cs.ModelRole.ORCHESTRATOR)

    def _get_default_cypher_config(self) -> ModelConfig:
        return self._get_default_config(cs.ModelRole.CYPHER)

    @property
    def active_orchestrator_config(self) -> ModelConfig:
        return self._active_orchestrator or self._get_default_orchestrator_config()

    @property
    def active_cypher_config(self) -> ModelConfig:
        return self._active_cypher or self._get_default_cypher_config()

    def set_orchestrator(
        self, provider: str, model: str, **kwargs: Unpack[ModelConfigKwargs]
    ) -> None:
        self._active_orchestrator = ModelConfig(
            provider=provider.lower(), model_id=model, **kwargs
        )

    def set_cypher(
        self, provider: str, model: str, **kwargs: Unpack[ModelConfigKwargs]
    ) -> None:
        self._active_cypher = ModelConfig(
            provider=provider.lower(), model_id=model, **kwargs
        )

    def parse_model_string(self, model_string: str) -> tuple[str, str]:
        if ":" not in model_string:
            return cs.Provider.OLLAMA, model_string
        provider, model = model_string.split(":", 1)
        if not provider:
            raise ValueError(ex.PROVIDER_EMPTY)
        return provider.lower(), model

    def resolve_batch_size(self, batch_size: int | None) -> int:
        resolved = self.MEMGRAPH_BATCH_SIZE if batch_size is None else batch_size
        if resolved < 1:
            raise ValueError(ex.BATCH_SIZE_POSITIVE)
        return resolved

    def get_project_mappings(self) -> dict[str, str]:
        """Load project mappings with priority:
        1. YAML configuration (.cgr_projects.yaml) if exists
        2. Fallback to single project (TARGET_REPO_PATH)

        Returns:
            Dictionary mapping project names to paths

        Examples:
            >>> # YAML exists
            >>> settings.get_project_mappings()
            {'gateway': '/path/to/gateway', 'logic': '/path/to/logic'}

            >>> # Single project mode
            >>> settings.TARGET_REPO_PATH = "/path/to/project"
            >>> settings.get_project_mappings()
            {'project': '/path/to/project'}
        """
        # Priority 1: Try YAML configuration
        yaml_path = Path(".cgr_projects.yaml")
        if yaml_path.exists():
            try:
                logger.info(f"Loading configuration from YAML: {yaml_path}")
                return self._load_yaml_config(yaml_path)
            except (ValueError, OSError) as e:
                logger.warning(
                    f"YAML configuration failed: {e}. Falling back to single project mode."
                )

        # Priority 2: Single project fallback
        logger.info("Using single project mode")
        return self._get_single_project_mapping()

    def _load_yaml_config(self, yaml_path: Path) -> dict[str, str]:
        """Load project mappings from YAML file

        Supports both simple dict and rich list formats

        Args:
            yaml_path: YAML configuration file path

        Returns:
            Dictionary mapping project names to paths

        Raises:
            ValueError: Invalid YAML format
        """
        try:
            import yaml
        except ImportError:
            raise ValueError(
                "PyYAML not installed, please run: uv add pyyaml"
            ) from None

        try:
            with yaml_path.open(encoding="utf-8") as f:
                config = yaml.safe_load(f) or {}

            projects = config.get("projects", {})

            # Auto-detect format
            if isinstance(projects, dict):
                # Simple format: {name: path}
                return self._validate_yaml_mappings(projects)
            elif isinstance(projects, list):
                # Rich format: [{name, path, ...}]
                return self._parse_rich_yaml_format(projects)
            else:
                raise ValueError(
                    f"Invalid 'projects' format. Expected dict or list, got: {type(projects)}"
                )
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing failed {yaml_path}: {e}")
            raise ValueError(f"Invalid YAML format: {e}") from e

    def _validate_yaml_mappings(self, mappings: dict) -> dict[str, str]:
        """Validate and normalize YAML mappings"""
        validated = {}
        for name, path in mappings.items():
            name = str(name).strip()
            path = str(path).strip()

            if not name or not path:
                logger.warning(f"Skipping invalid mapping: {name}:{path}")
                continue

            # Resolve to absolute path
            resolved_path = Path(path).resolve()
            if not resolved_path.exists():
                logger.warning(
                    f"Project path does not exist: {name} -> {resolved_path}"
                )

            validated[name] = str(resolved_path)

        if not validated:
            raise ValueError("No valid project mappings found in YAML")

        logger.info(f"Loaded {len(validated)} projects from YAML")
        return validated

    def _parse_rich_yaml_format(self, projects: list) -> dict[str, str]:
        """Parse rich YAML format (with metadata)"""
        mappings = {}
        for project in projects:
            if not isinstance(project, dict):
                logger.warning(f"Skipping invalid project entry: {project}")
                continue

            name = project.get("name")
            path = project.get("path")

            if not name or not path:
                logger.warning(f"Project missing name or path: {project}")
                continue

            # Record metadata for future use
            aliases = project.get("aliases", [])
            description = project.get("description", "")

            if aliases:
                logger.debug(f"Project '{name}' aliases: {aliases}")
            if description:
                logger.debug(f"Project '{name}' description: {description}")

            mappings[name] = str(Path(path).resolve())

        return mappings

    def _get_single_project_mapping(self) -> dict[str, str]:
        """Get single project mapping from TARGET_REPO_PATH"""
        default_path = Path(self.TARGET_REPO_PATH).resolve()
        default_project = default_path.name
        return {default_project: str(default_path)}


settings = AppConfig()

CGRIGNORE_FILENAME = ".cgrignore"


EMPTY_CGRIGNORE = CgrignorePatterns(exclude=frozenset(), unignore=frozenset())


def load_cgrignore_patterns(repo_path: Path) -> CgrignorePatterns:
    ignore_file = repo_path / CGRIGNORE_FILENAME
    if not ignore_file.is_file():
        return EMPTY_CGRIGNORE

    exclude: set[str] = set()
    unignore: set[str] = set()
    try:
        with ignore_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("!"):
                    unignore.add(line[1:].strip())
                else:
                    exclude.add(line)
        if exclude or unignore:
            logger.info(
                logs.CGRIGNORE_LOADED.format(
                    exclude_count=len(exclude),
                    unignore_count=len(unignore),
                    path=ignore_file,
                )
            )
        return CgrignorePatterns(
            exclude=frozenset(exclude),
            unignore=frozenset(unignore),
        )
    except OSError as e:
        logger.warning(logs.CGRIGNORE_READ_FAILED.format(path=ignore_file, error=e))
        return EMPTY_CGRIGNORE
