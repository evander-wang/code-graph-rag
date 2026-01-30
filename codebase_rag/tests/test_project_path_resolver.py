from __future__ import annotations

from pathlib import Path

import pytest

from codebase_rag.project_path_resolver import ProjectPathResolver


class TestProjectPathResolverInit:
    def test_init_with_mappings(self) -> None:
        mappings = {"project1": "/path/to/project1", "project2": "/path/to/project2"}
        resolver = ProjectPathResolver(mappings)

        assert resolver.list_projects() == ["project1", "project2"]
        assert (
            resolver.get_project_path("project1") == Path("/path/to/project1").resolve()
        )
        assert (
            resolver.get_project_path("project2") == Path("/path/to/project2").resolve()
        )

    def test_init_without_mappings_creates_default_entry(self) -> None:
        resolver = ProjectPathResolver()

        projects = resolver.list_projects()
        assert len(projects) >= 1


class TestExtractProjectName:
    def test_extracts_project_from_simple_fqn(self) -> None:
        resolver = ProjectPathResolver({"myproject": "/path"})
        result = resolver.extract_project_name("myproject.module.function")
        assert result == "myproject"

    def test_extracts_longest_matching_project(self) -> None:
        resolver = ProjectPathResolver(
            {"project": "/path", "project.sub": "/path2", "myproject": "/path3"}
        )
        result = resolver.extract_project_name("project.sub.module.function")
        assert result == "project.sub"

    def test_fallback_to_first_segment_when_no_match(self) -> None:
        resolver = ProjectPathResolver({"other": "/path"})
        result = resolver.extract_project_name("unknown.module.function")
        assert result == "unknown"

    def test_handles_fqn_with_dots(self) -> None:
        resolver = ProjectPathResolver({"project": "/path"})
        result = resolver.extract_project_name("project.package.module.Class.method")
        assert result == "project"

    def test_exact_match_over_prefix(self) -> None:
        resolver = ProjectPathResolver(
            {"myproject": "/path1", "myproject.sub": "/path2"}
        )
        result = resolver.extract_project_name("myproject.module.func")
        assert result == "myproject"

        result = resolver.extract_project_name("myproject.sub.module.func")
        assert result == "myproject.sub"


class TestGetProjectPath:
    def test_returns_path_for_existing_project(self) -> None:
        resolver = ProjectPathResolver({"myproject": "/tmp/test"})
        result = resolver.get_project_path("myproject")
        assert result == Path("/tmp/test").resolve()

    def test_raises_keyerror_for_nonexistent_project(self) -> None:
        resolver = ProjectPathResolver({"myproject": "/path"})
        with pytest.raises(KeyError, match="Project 'unknown' not found"):
            resolver.get_project_path("unknown")

    def test_error_message_includes_available_projects(self) -> None:
        resolver = ProjectPathResolver({"project1": "/path1", "project2": "/path2"})
        with pytest.raises(KeyError) as exc_info:
            resolver.get_project_path("unknown")

        error_msg = str(exc_info.value)
        assert "project1" in error_msg
        assert "project2" in error_msg


class TestResolvePathFromFqn:
    def test_resolves_path_from_qualified_name(self) -> None:
        resolver = ProjectPathResolver({"myproject": "/tmp/myproject"})
        result = resolver.resolve_path_from_fqn("myproject.module.function")
        assert result == Path("/tmp/myproject").resolve()

    def test_raises_keyerror_for_unknown_project_in_fqn(self) -> None:
        resolver = ProjectPathResolver({"known": "/path"})
        with pytest.raises(KeyError):
            resolver.resolve_path_from_fqn("unknown.module.function")


class TestListProjects:
    def test_with_mappings_returns_provided_projects(self) -> None:
        resolver = ProjectPathResolver(
            {"project1": "/path1", "project2": "/path2", "project3": "/path3"}
        )
        projects = resolver.list_projects()
        assert set(projects) == {"project1", "project2", "project3"}

    def test_returns_list_not_dict(self) -> None:
        resolver = ProjectPathResolver({"project": "/path"})
        result = resolver.list_projects()
        assert isinstance(result, list)
        assert not isinstance(result, dict)


class TestAddProject:
    def test_adds_new_project(self) -> None:
        resolver = ProjectPathResolver({})
        resolver.add_project("newproject", "/path/to/newproject")

        assert "newproject" in resolver.list_projects()
        assert (
            resolver.get_project_path("newproject")
            == Path("/path/to/newproject").resolve()
        )

    def test_adding_project_updates_existing(self) -> None:
        resolver = ProjectPathResolver({"myproject": "/old/path"})
        resolver.add_project("myproject", "/new/path")

        assert resolver.get_project_path("myproject") == Path("/new/path").resolve()

    def test_resolves_path_to_absolute(self) -> None:
        resolver = ProjectPathResolver({})
        resolver.add_project("myproject", "relative/path")

        result = resolver.get_project_path("myproject")
        assert result.is_absolute()


class TestRemoveProject:
    def test_removes_existing_project(self) -> None:
        resolver = ProjectPathResolver({"project1": "/path1", "project2": "/path2"})
        resolver.remove_project("project1")

        assert resolver.list_projects() == ["project2"]
        assert "project1" not in resolver.list_projects()

    def test_raises_keyerror_when_removing_nonexistent_project(self) -> None:
        resolver = ProjectPathResolver({"project": "/path"})
        with pytest.raises(KeyError, match="Project 'nonexistent' not found"):
            resolver.remove_project("nonexistent")

    def test_error_message_includes_available_projects_on_remove(self) -> None:
        resolver = ProjectPathResolver({"project1": "/path1", "project2": "/path2"})
        with pytest.raises(KeyError) as exc_info:
            resolver.remove_project("unknown")

        error_msg = str(exc_info.value)
        assert "project1" in error_msg
        assert "project2" in error_msg


class TestEdgeCases:
    def test_handles_empty_project_name(self) -> None:
        resolver = ProjectPathResolver({"": "/path"})
        result = resolver.extract_project_name(".module.function")
        assert result == ""

    def test_handles_fqn_without_dots(self) -> None:
        resolver = ProjectPathResolver({"myproject": "/path"})
        result = resolver.extract_project_name("myproject")
        assert result == "myproject"

    def test_path_resolution_with_symlinks(self, tmp_path: Path) -> None:
        target_dir = tmp_path / "target"
        target_dir.mkdir()
        link_dir = tmp_path / "link"
        link_dir.symlink_to(target_dir)

        resolver = ProjectPathResolver({"myproject": str(link_dir)})
        result = resolver.get_project_path("myproject")

        assert result.exists()
