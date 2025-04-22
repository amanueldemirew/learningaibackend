from typing import Dict, List, Any
from datetime import datetime


def convert_raw_toc_to_structured(
    raw_toc_data: Dict[str, Any], course_id: int, course_title: str
) -> Dict[str, Any]:
    """
    Convert raw TOC data to the structured format needed by the application.

    Args:
        raw_toc_data: The raw TOC data extracted from PDF
        course_id: The ID of the course
        course_title: The title of the course

    Returns:
        Structured TOC data in the format needed by the application
    """
    # Create the base structure
    structured_data = {
        "course_id": course_id,
        "course_title": course_title,
        "modules": [],
        "metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "source_file": "converted_toc.pdf",
        },
    }

    # Track seen module titles to avoid duplicates
    seen_titles = set()
    unique_modules = []

    # Process each module
    for module_idx, module in enumerate(raw_toc_data.get("modules", [])):
        module_title = module.get("title", f"Module {module_idx + 1}")

        # Extract page range from description if available
        description = module.get("description", "")
        page_range = module.get("page_range", "")

        # If page_range is available but not in description, add it to description
        if page_range and page_range not in description:
            if description:
                description = f"{description} (Pages: {page_range})"
            else:
                description = f"Pages: {page_range}"

        # Check if this module title has been seen before
        if module_title not in seen_titles:
            seen_titles.add(module_title)

            # Create a structured module
            structured_module = {
                "id": None,  # Will be set when saved to database
                "title": module_title,
                "description": description,
                "order": module.get("order", module_idx + 1),
                "units": [],
            }

            # Process each unit in the module
            for unit_idx, unit in enumerate(module.get("units", [])):
                # Extract page range from unit description if available
                unit_description = unit.get("description", "")
                unit_page_range = unit.get("page_range", "")

                # If unit_page_range is available but not in unit_description, add it to unit_description
                if unit_page_range and unit_page_range not in unit_description:
                    if unit_description:
                        unit_description = (
                            f"{unit_description} (Pages: {unit_page_range})"
                        )
                    else:
                        unit_description = f"Pages: {unit_page_range}"

                # Create a structured unit
                structured_unit = {
                    "id": None,  # Will be set when saved to database
                    "title": unit.get("title", f"Unit {unit_idx + 1}"),
                    "description": unit_description,
                    "order": unit.get("order", unit_idx + 1),
                    "content_generated": False,
                }

                # Add the unit to the module
                structured_module["units"].append(structured_unit)

            # Add the module to the structured data
            unique_modules.append(structured_module)
        else:
            # If it's a duplicate, merge its units with the existing module
            for existing_module in unique_modules:
                if existing_module["title"] == module_title:
                    # Process each unit in the duplicate module
                    for unit_idx, unit in enumerate(module.get("units", [])):
                        # Extract page range from unit description if available
                        unit_description = unit.get("description", "")
                        unit_page_range = unit.get("page_range", "")

                        # If unit_page_range is available but not in unit_description, add it to unit_description
                        if unit_page_range and unit_page_range not in unit_description:
                            if unit_description:
                                unit_description = (
                                    f"{unit_description} (Pages: {unit_page_range})"
                                )
                            else:
                                unit_description = f"Pages: {unit_page_range}"

                        # Create a structured unit
                        structured_unit = {
                            "id": None,  # Will be set when saved to database
                            "title": unit.get("title", f"Unit {unit_idx + 1}"),
                            "description": unit_description,
                            "order": unit.get("order", unit_idx + 1),
                            "content_generated": False,
                        }

                        # Check if this unit is already in the existing module
                        unit_exists = False
                        for existing_unit in existing_module["units"]:
                            if existing_unit["title"] == structured_unit["title"]:
                                unit_exists = True
                                break

                        # Add the unit to the module if it doesn't exist
                        if not unit_exists:
                            existing_module["units"].append(structured_unit)
                    break

    # Add the unique modules to the structured data
    structured_data["modules"] = unique_modules

    return structured_data


def convert_structured_toc_to_response(
    structured_data: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert structured TOC data to the response format.

    Args:
        structured_data: The structured TOC data

    Returns:
        TOC data in the response format
    """
    # The structured data is already in the response format
    # This function is a placeholder for any additional transformations
    # that might be needed in the future
    return structured_data


def process_toc_data(
    toc_data: Dict[str, Any], course_id: int, course_title: str
) -> Dict[str, Any]:
    """
    Process TOC data to ensure it's in the correct format.

    Args:
        toc_data: The TOC data to process
        course_id: The ID of the course
        course_title: The title of the course

    Returns:
        Processed TOC data in the correct format
    """
    # Check if the data is already in the structured format
    if "course_id" in toc_data and "course_title" in toc_data and "modules" in toc_data:
        # The data is already in the structured format
        return toc_data

    # Convert the raw TOC data to the structured format
    structured_data = convert_raw_toc_to_structured(toc_data, course_id, course_title)

    # Convert the structured data to the response format
    response_data = convert_structured_toc_to_response(structured_data)

    return response_data


def convert_example_format_to_structured(
    example_data: Dict[str, Any], course_id: int, course_title: str
) -> Dict[str, Any]:
    """
    Convert TOC data from the example format to the structured format.

    Example format:
    {
      "title": "Coordinate Geometry",
      "order": 7,
      "description": "It is found in page 1 - 50",
      "units": [
        {
          "title": "Distance between two points",
          "order": 1,
          "description": "It is found in page 1 - 50"
        },
        ...
      ]
    }

    Args:
        example_data: The TOC data in the example format
        course_id: The ID of the course
        course_title: The title of the course

    Returns:
        Structured TOC data in the format needed by the application
    """
    # Create the base structure
    structured_data = {
        "course_id": course_id,
        "course_title": course_title,
        "modules": [],
        "metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "source_file": "converted_toc.pdf",
        },
    }

    # Extract page range from description if available
    description = example_data.get("description", "")
    page_range = example_data.get("page_range", "")

    # If page_range is available but not in description, add it to description
    if page_range and page_range not in description:
        if description:
            description = f"{description} (Pages: {page_range})"
        else:
            description = f"Pages: {page_range}"

    # Create a module from the example data
    module = {
        "id": None,  # Will be set when saved to database
        "title": example_data.get("title", "Module"),
        "description": description,
        "order": example_data.get("order", 1),
        "units": [],
    }

    # Process each unit in the example data
    for unit_idx, unit in enumerate(example_data.get("units", [])):
        # Extract page range from unit description if available
        unit_description = unit.get("description", "")
        unit_page_range = unit.get("page_range", "")

        # If unit_page_range is available but not in unit_description, add it to unit_description
        if unit_page_range and unit_page_range not in unit_description:
            if unit_description:
                unit_description = f"{unit_description} (Pages: {unit_page_range})"
            else:
                unit_description = f"Pages: {unit_page_range}"

        # Create a structured unit
        structured_unit = {
            "id": None,  # Will be set when saved to database
            "title": unit.get("title", f"Unit {unit_idx + 1}"),
            "description": unit_description,
            "order": unit.get("order", unit_idx + 1),
            "content_generated": False,
        }

        # Add the unit to the module
        module["units"].append(structured_unit)

    # Add the module to the structured data
    structured_data["modules"].append(module)

    return structured_data


def process_example_toc_data(
    example_data: Dict[str, Any], course_id: int, course_title: str
) -> Dict[str, Any]:
    """
    Process TOC data in the example format to ensure it's in the correct format.

    Args:
        example_data: The TOC data in the example format
        course_id: The ID of the course
        course_title: The title of the course

    Returns:
        Processed TOC data in the correct format
    """
    # Convert the example data to the structured format
    structured_data = convert_example_format_to_structured(
        example_data, course_id, course_title
    )

    # Convert the structured data to the response format
    response_data = convert_structured_toc_to_response(structured_data)

    return response_data


def convert_multiple_example_modules_to_structured(
    example_modules: List[Dict[str, Any]], course_id: int, course_title: str
) -> Dict[str, Any]:
    """
    Convert multiple TOC modules from the example format to the structured format.

    Example format:
    [
      {
        "title": "Coordinate Geometry",
        "order": 7,
        "description": "It is found in page 1 - 50",
        "units": [
          {
            "title": "Distance between two points",
            "order": 1,
            "description": "It is found in page 1 - 50",
          },
          ...
        ]
      },
      ...
    ]

    Args:
        example_modules: A list of TOC modules in the example format
        course_id: The ID of the course
        course_title: The title of the course

    Returns:
        Structured TOC data in the format needed by the application
    """
    # Create the base structure
    structured_data = {
        "course_id": course_id,
        "course_title": course_title,
        "modules": [],
        "metadata": {
            "generated_at": datetime.utcnow().isoformat(),
            "source_file": "converted_toc.pdf",
        },
    }

    # Track seen module titles to avoid duplicates
    seen_titles = set()
    unique_modules = []

    # Process each module in the example data
    for module_idx, example_module in enumerate(example_modules):
        module_title = example_module.get("title", f"Module {module_idx + 1}")

        # Extract page range from description if available
        description = example_module.get("description", "")
        page_range = example_module.get("page_range", "")

        # If page_range is available but not in description, add it to description
        if page_range and page_range not in description:
            if description:
                description = f"{description} (Pages: {page_range})"
            else:
                description = f"Pages: {page_range}"

        # Check if this module title has been seen before
        if module_title not in seen_titles:
            seen_titles.add(module_title)

            # Create a structured module
            module = {
                "id": None,  # Will be set when saved to database
                "title": module_title,
                "description": description,
                "order": example_module.get("order", module_idx + 1),
                "units": [],
            }

            # Process each unit in the example module
            for unit_idx, unit in enumerate(example_module.get("units", [])):
                # Extract page range from unit description if available
                unit_description = unit.get("description", "")
                unit_page_range = unit.get("page_range", "")

                # If unit_page_range is available but not in unit_description, add it to unit_description
                if unit_page_range and unit_page_range not in unit_description:
                    if unit_description:
                        unit_description = (
                            f"{unit_description} (Pages: {unit_page_range})"
                        )
                    else:
                        unit_description = f"Pages: {unit_page_range}"

                # Create a structured unit
                structured_unit = {
                    "id": None,  # Will be set when saved to database
                    "title": unit.get("title", f"Unit {unit_idx + 1}"),
                    "description": unit_description,
                    "order": unit.get("order", unit_idx + 1),
                    "content_generated": False,
                }

                # Add the unit to the module
                module["units"].append(structured_unit)

            # Add the module to the structured data
            unique_modules.append(module)
        else:
            # If it's a duplicate, merge its units with the existing module
            for existing_module in unique_modules:
                if existing_module["title"] == module_title:
                    # Process each unit in the duplicate module
                    for unit_idx, unit in enumerate(example_module.get("units", [])):
                        # Extract page range from unit description if available
                        unit_description = unit.get("description", "")
                        unit_page_range = unit.get("page_range", "")

                        # If unit_page_range is available but not in unit_description, add it to unit_description
                        if unit_page_range and unit_page_range not in unit_description:
                            if unit_description:
                                unit_description = (
                                    f"{unit_description} (Pages: {unit_page_range})"
                                )
                            else:
                                unit_description = f"Pages: {unit_page_range}"

                        # Create a structured unit
                        structured_unit = {
                            "id": None,  # Will be set when saved to database
                            "title": unit.get("title", f"Unit {unit_idx + 1}"),
                            "description": unit_description,
                            "order": unit.get("order", unit_idx + 1),
                            "content_generated": False,
                        }

                        # Check if this unit is already in the existing module
                        unit_exists = False
                        for existing_unit in existing_module["units"]:
                            if existing_unit["title"] == structured_unit["title"]:
                                unit_exists = True
                                break

                        # Add the unit to the module if it doesn't exist
                        if not unit_exists:
                            existing_module["units"].append(structured_unit)
                    break

    # Add the unique modules to the structured data
    structured_data["modules"] = unique_modules

    return structured_data


def process_multiple_example_modules(
    example_modules: List[Dict[str, Any]], course_id: int, course_title: str
) -> Dict[str, Any]:
    """
    Process multiple TOC modules in the example format to ensure they're in the correct format.

    Args:
        example_modules: A list of TOC modules in the example format
        course_id: The ID of the course
        course_title: The title of the course

    Returns:
        Processed TOC data in the correct format
    """
    # Convert the example modules to the structured format
    structured_data = convert_multiple_example_modules_to_structured(
        example_modules, course_id, course_title
    )

    # Convert the structured data to the response format
    response_data = convert_structured_toc_to_response(structured_data)

    return response_data
