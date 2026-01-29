"""Multi-project path resolver component

This component maintains a mapping table of "project name → file system path",
supporting automatic extraction of project names from function qualified_names
and dynamic selection of the correct path.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from . import logs as ls

if TYPE_CHECKING:
    from .config import AppConfig


class ProjectPathResolver:
    """Project path resolver

    Maintains mappings from project names to file system paths,
    supporting path resolution in multi-project scenarios.
    """

    def __init__(self, mappings: dict[str, str] | None = None):
        """Initialize path resolver

        Args:
            mappings: Dictionary mapping project names to paths. If None,
                     loads default mappings from configuration.
        """
        self._mappings: dict[str, Path] = {}

        if mappings:
            # Use provided mappings
            for name, path in mappings.items():
                self._mappings[name] = Path(path).resolve()
            logger.info(
                ls.RESOLVER_INIT_MAPPED.format(
                    count=len(mappings), projects=list(mappings.keys())
                )
            )
        else:
            # Load default single-project mapping from config
            from .config import settings

            default_path = Path(settings.TARGET_REPO_PATH).resolve()
            default_project = default_path.name
            self._mappings[default_project] = default_path
            logger.info(
                ls.RESOLVER_INIT_DEFAULT.format(
                    project=default_project, path=default_path
                )
            )

    def extract_project_name(self, qualified_name: str) -> str:
        """Extract project name from qualified_name

        Uses longest prefix matching strategy to ensure correct matching
        when project names have hierarchical relationships.

        Args:
            qualified_name: Fully qualified name of the function, e.g., "user.profile.gateway.service.func"

        Returns:
            Extracted project name

        Examples:
            >>> resolver = ProjectPathResolver({
            ...     "user.profile.gateway": "/path/to/gateway",
            ...     "user.profile.logic": "/path/to/logic"
            ... })
            >>> resolver.extract_project_name("user.profile.gateway.service.Func")
            'user.profile.gateway'
            >>> resolver.extract_project_name("user.profile.logic.util.Helper")
            'user.profile.logic'
        """
        # Sort by length (longest first) to implement longest prefix matching
        sorted_projects = sorted(self._mappings.keys(), key=len, reverse=True)

        for project_name in sorted_projects:
            if qualified_name.startswith(project_name + "."):
                logger.debug(
                    ls.RESOLVER_EXTRACT_SUCCESS.format(
                        fqn=qualified_name, project=project_name
                    )
                )
                return project_name

        # If no match, use first segment as project name (backward compatibility)
        fallback = qualified_name.split(".")[0]
        logger.warning(
            ls.RESOLVER_EXTRACT_FALLBACK.format(fqn=qualified_name, fallback=fallback)
        )
        return fallback

    def get_project_path(self, project_name: str) -> Path:
        """Get file system path for a project

        Args:
            project_name: Project name

        Returns:
            File system path of the project

        Raises:
            KeyError: If project name doesn't exist in mapping table
        """
        if project_name not in self._mappings:
            available = ", ".join(self._mappings.keys())
            raise KeyError(
                ls.RESOLVER_PROJECT_NOT_FOUND.format(
                    project=project_name, available=available
                )
            )

        return self._mappings[project_name]

    def resolve_path_from_fqn(self, qualified_name: str) -> Path:
        """Resolve project path directly from qualified_name

        Args:
            qualified_name: Fully qualified name of the function

        Returns:
            File system path of the corresponding project
        """
        project_name = self.extract_project_name(qualified_name)
        return self.get_project_path(project_name)

    def list_projects(self) -> list[str]:
        """List all registered project names

        Returns:
            List of project names
        """
        return list(self._mappings.keys())

    def add_project(self, name: str, path: str) -> None:
        """Dynamically add a project mapping

        Args:
            name: Project name
            path: File system path
        """
        self._mappings[name] = Path(path).resolve()
        logger.info(ls.RESOLVER_PROJECT_ADDED.format(name=name, path=path))

    def remove_project(self, name: str) -> None:
        """Remove a project mapping

        Args:
            name: Project name

        Raises:
            KeyError: If project name doesn't exist
        """
        if name not in self._mappings:
            raise KeyError(
                ls.RESOLVER_PROJECT_NOT_FOUND.format(
                    project=name, available=", ".join(self._mappings.keys())
                )
            )
        del self._mappings[name]
        logger.info(ls.RESOLVER_PROJECT_REMOVED.format(name=name))

    @classmethod
    def from_config(cls, config: AppConfig) -> ProjectPathResolver:
        """Create resolver from configuration object

        Args:
            config: AppConfig configuration object

        Returns:
            ProjectPathResolver instance
        """
        mappings = config.get_project_mappings()
        return cls(mappings)
