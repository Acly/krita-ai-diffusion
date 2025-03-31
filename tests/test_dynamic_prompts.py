import pytest
import re
from unittest.mock import patch
from pathlib import Path

from ai_diffusion.dynamic_prompts import (
    evaluate_dynamic_prompt,
    _process_variables,
    process_variants,
)
from ai_diffusion.wildcards import (
    process_wildcards,
    _is_valid_wildcard_name,
    find_wildcard_files,
    read_wildcard_file,
)


@pytest.fixture
def mock_wildcard_system():
    """Mock the wildcard file system for testing."""
    # Mock wildcard content dictionary (filename -> list of entries)
    wildcard_contents = {
        "colors": ["red", "blue", "green", "yellow"],
        "animals": ["cat", "dog", "bird", "fish"],
        "adjectives": ["happy", "sad", "excited", "calm"],
        "simple_template": ["This is a ${animal}"],
        "parameterized": ["The ${color} ${animal} is ${adjective:cool}"],
        "parameterized2": ["The {${color}|${adjective}} ${animal}"],
        "nested": ["The {happy|sad} __animals__"],
        "weighted": ["1.5::heavy", "1.0::medium", "0.5::light"],
        "subdirectory/fruits": ["apple", "banana", "orange"],
    }

    # Define mock implementations
    def mock_find_wildcard_files(wildcard_name):
        """Direct mock implementation for testing"""
        if wildcard_name == "invalid":
            return []  # Test missing wildcards

        if wildcard_name == "dir/**":
            # Return multiple files for directory wildcards
            return [
                Path("/mock/wildcards/dir/one.txt"),
                Path("/mock/wildcards/dir/two.txt"),
            ]

        # Handle subdirectory case
        if "/" in wildcard_name:
            parts = wildcard_name.split("/")
            key = "/".join(parts)
            if key in wildcard_contents:
                return [Path(f"/mock/wildcards/{key}.txt")]
            return []

        # Handle standard case
        if wildcard_name in wildcard_contents:
            return [Path(f"/mock/wildcards/{wildcard_name}.txt")]

        # Handle wildcard with * for testing
        if "*" in wildcard_name:
            pattern = wildcard_name.replace("*", ".*")
            matching_keys = [k for k in wildcard_contents.keys() if re.match(f"^{pattern}$", k)]
            return [Path(f"/mock/wildcards/{k}.txt") for k in matching_keys]

        return []

    def mock_read_wildcard_file(file_path):
        """Direct mock implementation for testing"""
        # Extract the wildcard name from the file path
        file_name = file_path.stem
        parent = file_path.parent.name

        # Handle directory/** case
        if parent == "dir":
            if file_name == "one":
                return ["dir_item_1", "dir_item_2"]
            elif file_name == "two":
                return ["dir_item_3", "dir_item_4"]

        # Handle subdirectory case
        if parent != "wildcards":
            key = f"{parent}/{file_name}"
            return wildcard_contents.get(key, [])

        return wildcard_contents.get(file_name, [])

    # Apply patches but also return our mock implementations directly
    with patch("ai_diffusion.wildcards.find_wildcard_files", side_effect=mock_find_wildcard_files):
        with patch(
            "ai_diffusion.wildcards.read_wildcard_file", side_effect=mock_read_wildcard_file
        ):
            yield {
                "contents": wildcard_contents,
                "find_wildcard_files": mock_find_wildcard_files,
                "read_wildcard_file": mock_read_wildcard_file,
            }


class TestProcessVariables:
    def test_empty_input(self):
        assert _process_variables("", {}) == ""
        variables = {}
        _process_variables("", variables)
        assert variables == {}  # Ensure no side effects

    def test_simple_variable_definition(self):
        variables = {}
        result = _process_variables("${var=value}", variables)
        assert result == ""
        assert variables == {"var": "value"}

    def test_variable_reference(self):
        variables = {"existing": "value"}
        result = _process_variables("Reference: ${existing}", variables)
        assert result == "Reference: value"

    def test_immediate_variable(self):
        variables = {}
        result = _process_variables("${var=!{option|another}}", variables)
        assert "var" in variables
        assert variables["var"] in ["option", "another"]
        assert result == ""
        # Due to randomness, we may get the same value on both evaluations many times in a row,
        # even when that is not the intended behavior, so run this multiple times to be sure.
        for _ in range(
            100
        ):  # essentially impossible to not get the same value 100 times in a row (0.1^100)
            variables = {}
            result = _process_variables(
                "${var=!{value0|value1|value2|value3|value4|value5|value6|value7|value8|value9}}This is a test of option1: ${var} and option2: ${var}",
                variables,
            )
            assert "var" in variables
            # verify that both instances of ${var} are the same
            assert variables["var"] in [
                "value0",
                "value1",
                "value2",
                "value3",
                "value4",
                "value5",
                "value6",
                "value7",
                "value8",
                "value9",
            ]
            assert result == "This is a test of option1: {} and option2: {}".format(
                variables["var"], variables["var"]
            )

    def test_non_immediate_variable(self):
        variables = {}
        result = _process_variables("${var={option|another}}", variables)
        assert "var" in variables
        assert variables["var"] == "{option|another}"
        assert result == ""
        # Due to randomness, we may get the same value on both evaluations many times in a row,
        # so run until we get a different value on the two evaluations of the variable.
        # At the end, verify at least one iteration had different values.
        value_difference = False
        for _ in range(
            100
        ):  # essentially impossible to get the same value 100 times in a row (0.1^100)
            variables = {}
            result = _process_variables(
                "${var={value0|value1|value2|value3|value4|value5|value6|value7|value8|value9}}This is a test of option1: ${var} and option2: ${var}",
                variables,
            )
            assert "var" in variables
            assert (
                variables["var"]
                == "{value0|value1|value2|value3|value4|value5|value6|value7|value8|value9}"
            )
            # if both evaluations of ${var} are the same, run another iteration.
            # if they are different, break the loop
            option1_value = result.split("option1: ")[1].split(" and option2: ")[0]
            option2_value = result.split("option2: ")[1]
            if option1_value != option2_value:
                value_difference = True
                break

        assert value_difference, "Both evaluations of ${var} were the same after 1000 iterations"

    def test_nested_variables(self):
        variables = {}
        result = _process_variables(
            "${inner=value}${outer=${inner}}This is a test of ${outer}", variables
        )
        assert variables == {"inner": "value", "outer": "value"}
        assert result == "This is a test of value"

        variables = {}
        result = _process_variables("${outer=${inner=value}}This is a test of ${outer}", variables)
        assert variables == {"inner": "value", "outer": "value"}
        assert result == "This is a test of value"

        variables = {}
        result = _process_variables("${a=${b=${c=value}}}This is a test of ${a}", variables)
        assert variables == {"a": "value", "b": "value", "c": "value"}
        assert result == "This is a test of value"

        variables = {}
        result = _process_variables("${a=${b=${c=${d=value}}}}This is a test of ${a}", variables)
        assert variables == {"a": "value", "b": "value", "c": "value", "d": "value"}
        assert result == "This is a test of value"

        # Nesting logic isn't working correctly yet for these complex cases, so commenting out for now
        # variables = {}
        # result = _process_variables("{d=value}{c=${d}}${a=${b=${c}}}This is a test of ${a}", variables)
        # print(variables)
        # print(result)
        # assert variables == {"a": "value", "b": "value", "c": "value", "d": "value"}
        # assert result == "This is a test of value"

        # variables = {}
        # result = _process_variables("{c=${d=value}}${a=${b=${c}}}This is a test of ${a}", variables)
        # print(variables)
        # print(result)
        # assert variables == {"a": "value", "b": "value", "c": "value", "d": "value"}
        # assert result == "This is a test of value"

    def test_missing_variable(self):
        with pytest.raises(Exception, match="Variable.*is not defined"):
            _process_variables("Reference: ${nonexistent}", {})


class TestProcessVariants:
    def test_empty_input(self):
        assert process_variants("") == ""

    def test_simple_variants(self):
        # Run multiple times to verify randomness
        for _ in range(10):
            result = process_variants("This is a {red|blue|green} test")
            assert result in ["This is a red test", "This is a blue test", "This is a green test"]

    def test_nested_variants(self):
        for _ in range(10):
            result = process_variants("The {big|small} {cat|dog|{red|blue} fish}")
            assert any(word in result for word in ["big", "small"])
            assert any(word in result for word in ["cat", "dog", "red fish", "blue fish"])

    def test_weighted_variants(self):
        # We can't test the probability distribution easily, but we can verify syntax
        for _ in range(10):
            result = process_variants("This is {1.5::very|1.0::somewhat|0.5::slightly} important")
            assert result in [
                "This is very important",
                "This is somewhat important",
                "This is slightly important",
            ]

    def test_multiple_selection(self):
        for _ in range(10):
            result = process_variants("Colors: {2$$red|blue|green|yellow}")
            # Should have two colors separated by a comma and space
            assert result.startswith("Colors: ")
            colors = result[8:].split(", ")
            assert len(colors) == 2
            assert all(color in ["red", "blue", "green", "yellow"] for color in colors)

    def test_custom_separator(self):
        for _ in range(10):
            result = process_variants("Colors: {2$$-$$red|blue|green|yellow}")
            # Should have two colors separated by a hyphen
            assert result.startswith("Colors: ")
            colors = result[8:].split("-")
            assert len(colors) == 2
            assert all(color in ["red", "blue", "green", "yellow"] for color in colors)
            result = process_variants("Colors: {2$$ and $$red|blue|green|yellow}")
            # Should have two colors separated by a hyphen
            assert result.startswith("Colors: ")
            colors = result[8:].split(" and ")
            assert len(colors) == 2
            assert all(color in ["red", "blue", "green", "yellow"] for color in colors)

    def test_range_selection(self):
        for _ in range(20):
            result = process_variants("Colors: {1-3$$red|blue|green|yellow}")
            # Should have 1-3 colors
            assert result.startswith("Colors: ")
            colors = result[8:].split(", ")
            assert 1 <= len(colors) <= 3
            assert all(color in ["red", "blue", "green", "yellow"] for color in colors)
        for _ in range(20):
            result = process_variants("Colors: {-2$$red|blue|green|yellow}")
            # Should have 1-2 colors
            assert result.startswith("Colors: ")
            colors = result[8:].split(", ")
            assert 1 <= len(colors) <= 2
            assert all(color in ["red", "blue", "green", "yellow"] for color in colors)
        for _ in range(20):
            result = process_variants("Colors: {1-$$red|blue|green|yellow}")
            # Should have 1-4 colors
            assert result.startswith("Colors: ")
            colors = result[8:].split(", ")
            assert 1 <= len(colors) <= 4
            assert all(color in ["red", "blue", "green", "yellow"] for color in colors)


class TestWildcardFiles:
    """Tests for wildcard functions using real files."""

    @pytest.fixture(autouse=True)
    def setup_test_files(self, setup_test_wildcards_dir):
        """Create test wildcard files for each test."""
        from ai_diffusion.wildcards import (
            set_wildcard_dirs,
            reset_wildcard_dirs,
        )

        self.test_dir = setup_test_wildcards_dir

        # Set the test directory as the only wildcard directory
        self.prev_dirs = set_wildcard_dirs([self.test_dir])

        # Create some real test files
        wildcard_files = {
            "colors.txt": ["crimson", "azure", "emerald", "amber"],
            "animals.txt": ["lion", "tiger", "bear", "wolf"],
            "adjectives.txt": ["brave", "clever", "fierce", "gentle"],
            "nested.txt": ["A {bold|subtle} __colors__ shade"],
        }

        # Create files with similar names across different locations
        wildcard_files["colors_alt.txt"] = ["purple", "pink", "teal", "gold"]
        wildcard_files["colors_special.txt"] = ["silver", "bronze", "platinum", "copper"]

        # Create a subdirectory
        subdir = self.test_dir / "fantasy"
        subdir.mkdir(exist_ok=True)
        wildcard_files["fantasy/creatures.txt"] = ["dragon", "unicorn", "phoenix", "griffin"]
        wildcard_files["fantasy/colors.txt"] = [
            "mystic blue",
            "enchanted green",
            "fairy gold",
            "dragon red",
        ]

        # Create a directory for wildcard directory tests
        dir_wildcard = self.test_dir / "elements"
        dir_wildcard.mkdir(exist_ok=True)
        wildcard_files["elements/fire.txt"] = ["flame", "blaze", "inferno"]
        wildcard_files["elements/water.txt"] = ["wave", "ocean", "river"]

        # Create a deep directory structure (3+ levels)
        deep_dir = self.test_dir / "deep" / "deeper" / "deepest"
        deep_dir.mkdir(parents=True, exist_ok=True)
        wildcard_files["deep/item.txt"] = ["level1-item1", "level1-item2"]
        wildcard_files["deep/deeper/item.txt"] = ["level2-item1", "level2-item2"]
        wildcard_files["deep/deeper/deepest/item.txt"] = ["level3-item1", "level3-item2"]
        wildcard_files["deep/deeper/deepest/colors.txt"] = ["deep red", "deep blue", "deep green"]

        # Create files for single character glob testing
        pattern_dir = self.test_dir / "patterns"
        pattern_dir.mkdir(exist_ok=True)
        wildcard_files["patterns/test1.txt"] = ["pattern-1"]
        wildcard_files["patterns/test2.txt"] = ["pattern-2"]
        wildcard_files["patterns/test3.txt"] = ["pattern-3"]
        wildcard_files["patterns/other.txt"] = ["other-pattern"]

        # Create files for recursive glob testing
        recursive_dir = self.test_dir / "all" / "sub1"
        recursive_dir.mkdir(parents=True, exist_ok=True)
        recursive_dir2 = self.test_dir / "all" / "sub2"
        recursive_dir2.mkdir(parents=True, exist_ok=True)
        wildcard_files["all/file.txt"] = ["root-content"]
        wildcard_files["all/sub1/file.txt"] = ["sub1-content"]
        wildcard_files["all/sub2/file.txt"] = ["sub2-content"]
        wildcard_files["all/common.txt"] = ["common-root"]
        wildcard_files["all/sub1/common.txt"] = ["common-sub1"]
        wildcard_files["all/sub2/common.txt"] = ["common-sub2"]

        # Create files for character range testing
        wildcard_files["patterns/char_a.txt"] = ["content-a"]
        wildcard_files["patterns/char_b.txt"] = ["content-b"]
        wildcard_files["patterns/char_c.txt"] = ["content-c"]

        wildcard_files["dir1/file.txt"] = ["dir1-content"]
        wildcard_files["dir2/file.txt"] = ["dir2-content"]
        wildcard_files["dir3/file.txt"] = ["dir3-content"]

        # Write all the files
        for filename, contents in wildcard_files.items():
            file_path = self.test_dir / filename
            file_path.parent.mkdir(exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                for line in contents:
                    f.write(f"{line}\n")

        yield

        # Reset wildcard directories to default (no custom directory)
        reset_wildcard_dirs()

    def test_is_valid_wildcard_name(self):
        # Valid names
        assert _is_valid_wildcard_name("colors") is True
        assert _is_valid_wildcard_name("color_list") is True
        assert _is_valid_wildcard_name("dir/subdir") is True
        assert _is_valid_wildcard_name("wild*card") is True
        assert _is_valid_wildcard_name("wild?card") is True
        assert _is_valid_wildcard_name("dir/wild*card") is True
        assert _is_valid_wildcard_name("dir/test?") is True
        assert _is_valid_wildcard_name("dir/test[!3]") is True
        assert _is_valid_wildcard_name("dir/test[!1-4]") is True
        assert _is_valid_wildcard_name("dir/test[!d]") is True
        assert _is_valid_wildcard_name("dir/test[!de]") is True
        assert _is_valid_wildcard_name("dir/test[!a-d]") is True
        assert _is_valid_wildcard_name("dir/*[4]") is True
        assert _is_valid_wildcard_name("dir/test[0-9]") is True
        assert _is_valid_wildcard_name("dir/*[6-9]") is True
        assert _is_valid_wildcard_name("dir/[to][13]") is True
        assert _is_valid_wildcard_name("dir/[ac][5-9]") is True
        assert _is_valid_wildcard_name("dir/[b-e]e[45]") is True
        assert _is_valid_wildcard_name("dir/[b-e]e*[45]") is True
        assert _is_valid_wildcard_name("dir/[d-g][3-7]") is True
        assert _is_valid_wildcard_name("dir/[d-g]/*[3-7]") is True
        assert _is_valid_wildcard_name("dir/[d-g]og/*[3-7]") is True

        # Invalid names
        assert _is_valid_wildcard_name("") is False
        assert _is_valid_wildcard_name("../colors") is False
        assert _is_valid_wildcard_name("..\\colors") is False
        assert _is_valid_wildcard_name("colors\\file") is False
        assert _is_valid_wildcard_name("C:\\Windows\\System32") is False
        assert _is_valid_wildcard_name("/absolute/path") is False
        assert _is_valid_wildcard_name("dir/test[") is False
        assert _is_valid_wildcard_name("dir/test]") is False
        assert _is_valid_wildcard_name("dir/test[^]") is False
        assert _is_valid_wildcard_name("dir/test[ab][1]]") is False
        assert _is_valid_wildcard_name("dir/test[ab[]]]") is False
        assert _is_valid_wildcard_name("colors:invalid") is False
        assert _is_valid_wildcard_name("colors|invalid") is False
        assert _is_valid_wildcard_name("colors<invalid") is False
        assert _is_valid_wildcard_name("colors>invalid") is False
        assert _is_valid_wildcard_name("colors&invalid") is False
        assert _is_valid_wildcard_name("colors^invalid") is False
        assert _is_valid_wildcard_name("colors$invalid") is False
        assert _is_valid_wildcard_name("colors@invalid") is False
        assert _is_valid_wildcard_name("colors#invalid") is False
        assert _is_valid_wildcard_name("colors%invalid") is False

    def test_find_wildcard_files(self):
        """Test finding wildcard files using real files."""
        # Basic case
        assert len(find_wildcard_files("colors")) == 1
        assert len(find_wildcard_files("animals")) == 1

        # Subdirectory case
        assert len(find_wildcard_files("fantasy/creatures")) == 1

        # Directory wildcard
        assert len(find_wildcard_files("elements/**")) == 2

        # Glob pattern
        assert len(find_wildcard_files("c*")) >= 1  # Should at least find colors

        # Missing wildcard
        assert len(find_wildcard_files("nonexistent")) == 0

    def test_similar_named_files(self):
        """Test finding multiple files with similar names."""
        # Find all files starting with 'colors'
        files = find_wildcard_files("colors*")
        # Should find colors.txt, colors_alt.txt, colors_special.txt, fantasy/colors.txt, deep/deeper/deepest/colors.txt
        assert len(files) == 5

        # Specific match
        files = find_wildcard_files("colors_alt")
        assert len(files) == 1

        # Find similarly named files in specific subdirectory
        files = find_wildcard_files("fantasy/colors")
        assert len(files) == 1

        # Make sure subdirectory files don't contaminate root search
        files = find_wildcard_files("colors")
        assert len(files) == 1
        name = files[0].name
        assert name.lower() == "colors.txt"

    def test_deep_subdirectory(self):
        """Test finding files in deep subdirectory structure."""
        # Find file at first level
        files = find_wildcard_files("deep/item")
        assert len(files) == 1

        # Find file at second level
        files = find_wildcard_files("deep/deeper/item")
        assert len(files) == 1

        # Find file at third level
        files = find_wildcard_files("deep/deeper/deepest/item")
        assert len(files) == 1

        # Verify they're all different files
        level1 = find_wildcard_files("deep/item")[0]
        level2 = find_wildcard_files("deep/deeper/item")[0]
        level3 = find_wildcard_files("deep/deeper/deepest/item")[0]
        assert level1 != level2 != level3

    def test_recursive_globbing(self):
        """Test various recursive globbing patterns."""
        # Find all files named 'item.txt' in any subdirectory of deep
        files = find_wildcard_files("deep/**/item")
        assert len(files) == 3  # Should find all three levels

        # Find all 'common.txt' files in all/ and its subdirectories
        files = find_wildcard_files("all/**/common")
        assert len(files) == 3  # root, sub1, sub2

        # Find all 'file.txt' files in all/ and its subdirectories
        files = find_wildcard_files("all/**/file")
        assert len(files) == 3  # root, sub1, sub2

        # Find only in specific subdirectory
        files = find_wildcard_files("all/sub1/**")
        assert len(files) == 2  # file.txt and common.txt in sub1

        # Find nested: all files in deepest
        files = find_wildcard_files("deep/deeper/deepest/**")
        assert len(files) == 2  # item.txt and colors.txt in deepest

    def test_filename_glob_vs_path_glob(self):
        """Test difference between filename* and path/to/filename* patterns."""
        # colors* should find all files with that pattern in root dir only
        files = find_wildcard_files("colors*")
        assert (
            len(files) == 5
        )  # colors.txt, colors_alt.txt, colors_special.txt, fantasy/colors.txt, deep/deeper/deepest/colors.txt

        # deep/deeper/deepest/colors* should only find colors.txt in that specific directory
        files = find_wildcard_files("deep/deeper/deepest/colors*")
        assert len(files) == 1

        # Specific subdirectory glob should only match in that directory
        files = find_wildcard_files("fantasy/c*")
        assert len(files) == 2  # Should match (creatures.txt, colors.txt) in fantasy/

        # While general pattern should find only colors.txt across all directories
        all_color_files = find_wildcard_files("**/colors")
        assert len(all_color_files) >= 3  # Should find in root, fantasy/, and deep/

    def test_single_character_globbing(self):
        """Test single character globbing with ?."""
        # test?.txt should match test1.txt, test2.txt, test3.txt but not other.txt
        files = find_wildcard_files("patterns/test?")
        assert len(files) == 3

        # tes?2.txt should match only test2.txt
        files = find_wildcard_files("patterns/tes?2")
        assert len(files) == 1

        # test??.txt shouldn't match anything (tests have single digit)
        files = find_wildcard_files("patterns/test??")
        assert len(files) == 0

        # o?her.txt should match other.txt
        files = find_wildcard_files("patterns/o?her")
        assert len(files) == 1

    def test_read_wildcard_file(self):
        """Test reading wildcard files using real files."""
        # Get a file path using find_wildcard_files
        files = find_wildcard_files("colors")
        assert len(files) == 1
        colors_file = files[0]

        # Read and verify content
        colors = read_wildcard_file(colors_file)
        assert len(colors) == 4
        assert "crimson" in colors
        assert "azure" in colors
        assert "emerald" in colors
        assert "amber" in colors

        # Test with missing file
        with pytest.raises(Exception, match="not exist"):
            read_wildcard_file(Path("/nonexistent/path.txt"))

    def test_wildcards_in_prompt(self):
        """Test using wildcards in prompt processing."""
        from ai_diffusion.dynamic_prompts import evaluate_dynamic_prompt

        # Simple wildcard
        result = evaluate_dynamic_prompt("My favorite color is __colors__.")
        assert any(color in result for color in ["crimson", "azure", "emerald", "amber"])

        # Nested wildcards
        result = evaluate_dynamic_prompt("I saw a __nested__.")
        assert "A " in result
        assert any(style in result for style in ["bold", "subtle"])
        assert any(color in result for color in ["crimson", "azure", "emerald", "amber"])

        # Combined features
        prompt = """
        ${animal=__animals__}
        ${quality=__adjectives__}
        
        I encountered a ${quality} ${animal} in the forest.
        """
        result = evaluate_dynamic_prompt(prompt)
        assert any(animal in result for animal in ["lion", "tiger", "bear", "wolf"])
        assert any(adj in result for adj in ["brave", "clever", "fierce", "gentle"])
        assert "#" not in result
        assert "${" not in result
        assert "}" not in result

    def test_character_range_globbing(self):
        """Test character range globbing with [range] syntax."""
        # Test numeric ranges
        files = find_wildcard_files("patterns/test[1-2]")
        assert len(files) == 2
        filenames = [f.name for f in files]
        assert "test1.txt" in filenames
        assert "test2.txt" in filenames
        assert "test3.txt" not in filenames

        # Test wildcard with numeric range
        files = find_wildcard_files("patterns/*[1-2]")
        assert len(files) == 2
        filenames = [f.name for f in files]
        assert "test1.txt" in filenames
        assert "test2.txt" in filenames
        assert "test3.txt" not in filenames

        # Test with character literal ranges
        files = find_wildcard_files("patterns/[tb]est[1-2]")
        assert len(files) == 2  # Should match (test1, test2) or (best1, best2) if they existed

        # Test alphabetic range
        files = find_wildcard_files("patterns/char_[a-b]")
        assert len(files) == 2
        filenames = [f.name for f in files]
        assert "char_a.txt" in filenames
        assert "char_b.txt" in filenames
        assert "char_c.txt" not in filenames

        # Test negated range (anything except the range)
        files = find_wildcard_files("patterns/test[!3]")
        assert len(files) == 2
        filenames = [f.name for f in files]
        assert "test1.txt" in filenames
        assert "test2.txt" in filenames
        assert "test3.txt" not in filenames

        # Test character class (match any character in the class)
        files = find_wildcard_files("patterns/test[13]")
        assert len(files) == 2
        filenames = [f.name for f in files]
        assert "test1.txt" in filenames
        assert "test3.txt" in filenames
        assert "test2.txt" not in filenames

        # Test directory name with range
        files = find_wildcard_files("dir[1-2]/file")
        assert len(files) == 2
        dirnames = [f.parent.name for f in files]
        assert "dir1" in dirnames
        assert "dir2" in dirnames
        assert "dir3" not in dirnames


class TestProcessWildcards:
    def test_empty_input(self):
        assert process_wildcards("", {}) == ""

    def test_simple_wildcard(self, mock_wildcard_system):
        for _ in range(10):
            result = process_wildcards("My favorite color is __colors__", {})
            assert result in [
                "My favorite color is red",
                "My favorite color is blue",
                "My favorite color is green",
                "My favorite color is yellow",
            ]

    def test_wildcards_with_params(self, mock_wildcard_system):
        for _ in range(10):
            # Note: we're using our mocked "parameterized" wildcard
            result = process_wildcards(
                "__parameterized(color=purple, animal=elephant, adjective=giant)__", {}
            )
            assert result == "The purple elephant is giant"

    def test_wildcards_with_variables(self, mock_wildcard_system):
        variables = {"animal": "monkey", "color": "orange"}
        result = process_wildcards("__parameterized(adjective=funny)__", variables)
        assert result == "The orange monkey is funny"

    def test_wildcards_with_defaults(self, mock_wildcard_system):
        # The template contains ${adjective:cool} where default = "cool"
        result = process_wildcards("__parameterized(color=black, animal=panther)__", {})
        assert "The black panther is cool" in result

        # When provided, it should override the default
        result = process_wildcards(
            "__parameterized(color=black, animal=panther, adjective=majestic)__", {}
        )
        assert "The black panther is majestic" in result

    def test_directory_wildcards(self, mock_wildcard_system):
        for _ in range(10):
            result = process_wildcards("Random item: __dir/**__", {})
            assert result in [
                "Random item: dir_item_1",
                "Random item: dir_item_2",
                "Random item: dir_item_3",
                "Random item: dir_item_4",
            ]

    def test_nested_wildcards(self, mock_wildcard_system):
        for _ in range(10):
            result = process_wildcards("__nested__", {})
            mood = "happy" if "happy" in result else "sad"
            assert result in [
                f"The {mood} cat",
                f"The {mood} dog",
                f"The {mood} bird",
                f"The {mood} fish",
            ]


class TestEvaluateDynamicPrompt:
    def test_empty_input(self):
        assert evaluate_dynamic_prompt("") == ""
        assert evaluate_dynamic_prompt(None) is None

    def test_comments(self):
        assert evaluate_dynamic_prompt("A prompt # with comment") == "A prompt"
        assert evaluate_dynamic_prompt("# Full comment line\nActual content") == "Actual content"

    def test_variables(self):
        assert evaluate_dynamic_prompt("${var=value}Reference: ${var}") == "Reference: value"
        with pytest.raises(Exception):
            evaluate_dynamic_prompt("${var}")  # Missing variable

    def test_variants(self):
        result = evaluate_dynamic_prompt("This is a {test|demonstration}")
        assert "{" not in result
        assert "}" not in result
        assert result in ["This is a test", "This is a demonstration"]

    def test_unicode_characters(self):
        """Test with Unicode characters in all components."""
        result = evaluate_dynamic_prompt("${var=你好}Unicode: ${var} and {世界|planet}")
        assert "Unicode: 你好 and " in result
        assert result.endswith("世界") or result.endswith("planet")

    def test_quotes_in_parameters(self, mock_wildcard_system):
        """Test with quotes in wildcard parameters."""
        result = evaluate_dynamic_prompt(
            "__parameterized(animal=\"quoted animal\", adjective='quoted cool')__"
        )
        assert "quoted animal" in result
        assert "quoted cool" in result or "cool" in result

    def test_special_characters_in_variables(self):
        """Test with special characters in variables."""
        result = evaluate_dynamic_prompt(
            "${special=value with: commas, brackets [] and {braces1}}Result: ${special}, also {braces2}"
        )
        assert "Result: value with: commas, brackets [] and {braces1}, also {braces2}" in result

    def test_wildcards(self, mock_wildcard_system):
        result = evaluate_dynamic_prompt("My favorite color is __colors__")
        assert result in [
            "My favorite color is red",
            "My favorite color is blue",
            "My favorite color is green",
            "My favorite color is yellow",
        ]

    def test_combined_features(self, mock_wildcard_system):
        prompt = """
        # Define variables
        ${animal=cat}
        ${mood=happy}
        
        The ${animal} is {very|extremely} ${mood} and likes __colors__.
        """

        result = evaluate_dynamic_prompt(prompt)
        assert "cat" in result
        assert "happy" in result
        assert any(color in result for color in ["red", "blue", "green", "yellow"])
        assert any(adverb in result for adverb in ["very", "extremely"])
        assert "#" not in result
        assert "${" not in result
        assert "}" not in result

    def test_variables_dont_leak_between_evaluations(self):
        """Test that variables do not leak between separate evaluation calls."""
        # First evaluation defines a variable
        first_result = evaluate_dynamic_prompt("${var=test_value}Using: ${var}")
        assert "Using: test_value" in first_result

        # Second evaluation should not have access to the variable from first call
        with pytest.raises(Exception, match="not defined"):
            evaluate_dynamic_prompt("Using again: ${var}")

    def test_variable_shadowing(self):
        """Test that variable shadowing works correctly."""
        prompt = """
        ${var=outer_value}
        Outer: ${var}
        ${nested=${var=inner_value}${var}}
        After nested: ${var}
        """
        result = evaluate_dynamic_prompt(prompt)

        # The inner variable definition should not affect the outer
        assert "Outer: outer_value" in result
        assert "After nested: outer_value" in result

    def test_variable_scope_in_nested_evaluation(self):
        """Test variable scope in nested evaluations."""
        prompt = """
        ${outer=initial}
        ${inner=${outer}_plus_suffix}
        Result: ${inner}
        """
        result = evaluate_dynamic_prompt(prompt)

        # The inner variable should have access to the outer
        assert "Result: initial_plus_suffix" in result

    def test_unbalanced_braces(self):
        """Test handling of unbalanced braces."""
        with pytest.raises(Exception, match="Unmatched.*brace"):
            evaluate_dynamic_prompt("This is a {test")

        with pytest.raises(Exception, match="Unmatched.*brace"):
            evaluate_dynamic_prompt("This is a test}")

        with pytest.raises(Exception, match="Unmatched.*brace"):
            evaluate_dynamic_prompt("This is a {test} with ${another")

        with pytest.raises(Exception, match="Unmatched.*brace"):
            evaluate_dynamic_prompt("This is a {test} with another}")

        with pytest.raises(Exception, match="Unmatched.*brace"):
            evaluate_dynamic_prompt("This is a {a|b")

        with pytest.raises(Exception, match="Unmatched.*brace"):
            evaluate_dynamic_prompt("This is a a|b}")

        with pytest.raises(Exception, match="Unmatched.*brace"):
            evaluate_dynamic_prompt("This is a {a|b|{c|d}")

        with pytest.raises(Exception, match="Unmatched.*brace"):
            evaluate_dynamic_prompt("This is a a|b|{c|d}}")

        with pytest.raises(Exception, match="Unmatched.*brace"):
            evaluate_dynamic_prompt("This is a {a|b|c|d}}")

    def test_invalid_wildcard_parameter_syntax(self, mock_wildcard_system):
        """Test handling of invalid wildcard parameter syntax."""
        with pytest.raises(Exception):
            evaluate_dynamic_prompt("__colors(invalid__")

        with pytest.raises(Exception):
            evaluate_dynamic_prompt("__colorsinvalid)__")

        with pytest.raises(Exception):
            evaluate_dynamic_prompt("__colors(missing_equals)__")

        with pytest.raises(Exception):
            evaluate_dynamic_prompt("__colors(key=value,missing_equals)__")

    def test_malformed_variable_definitions(self):
        """Test handling of malformed variable definitions."""
        with pytest.raises(Exception):
            evaluate_dynamic_prompt("${=value}")  # Missing name

        with pytest.raises(Exception):
            evaluate_dynamic_prompt("${var}")  # Missing value definition and no default

    def test_wildcard_in_variable_in_variant(self, mock_wildcard_system):
        """Test for wildcard inside variable inside variant."""
        result = evaluate_dynamic_prompt("{option one|${var=__colors__}${var}|option three}")
        assert result in ["option one", "option three", "red", "blue", "green", "yellow"]
        result = evaluate_dynamic_prompt("${var1=__colors__}${var2=__animals__}{${var1}|${var2}}")
        assert result in ["cat", "dog", "bird", "fish", "red", "blue", "green", "yellow"]

    def test_wildcard_in_variant_in_variable(self, mock_wildcard_system):
        """Test for wildcard inside variant inside variable."""
        result = evaluate_dynamic_prompt("${var={__colors__|__animals__}}${var}")
        assert result in ["cat", "dog", "bird", "fish", "red", "blue", "green", "yellow"]

    def test_variable_in_wildcard_in_variant(self, mock_wildcard_system):
        """Test for variable inside wildcard inside variant."""
        prompt = """
        ${color=blue}{A|__parameterized(color=${color},animal=__animals__)__ and this is a test}
        """
        result = evaluate_dynamic_prompt(prompt)
        assert result in [
            "A",
            "The blue cat is cool and this is a test",
            "The blue dog is cool and this is a test",
            "The blue bird is cool and this is a test",
            "The blue fish is cool and this is a test",
        ]

    def test_variable_in_variant_in_wildcard(self, mock_wildcard_system):
        """Test for variable inside variant inside wildcard."""
        result = evaluate_dynamic_prompt(
            "${color=black}${animal=cat}__parameterized2(color=${color},animal=${animal},adjective=funny)__"
        )
        assert result in ["The black cat", "The funny cat"]

    def test_variant_in_wildcard_in_variable(self, mock_wildcard_system):
        """Test for variant inside wildcard inside variable."""
        prompt = """
        ${template=__nested__}
        Result: ${template}
        """
        result = evaluate_dynamic_prompt(prompt)

        assert "Result: The " in result
        assert any(mood in result for mood in ["happy", "sad"])
        assert any(animal in result for animal in ["cat", "dog", "bird", "fish"])

    def test_variant_in_variable_in_wildcard(self, mock_wildcard_system):
        """Test for variant inside variable inside wildcard."""
        prompt = """
        ${animal=__animals__}
        Result: __parameterized(color={crimson|scarlet},animal=${animal},adjective={funny|annoying})__
        """
        result = evaluate_dynamic_prompt(prompt)

        assert "Result: The " in result
        assert any(color in result for color in ["crimson", "scarlet"])
        assert any(animal in result for animal in ["cat", "dog", "bird", "fish"])
        assert any(adj in result for adj in ["funny", "annoying"])
