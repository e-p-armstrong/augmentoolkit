def resolve_path(path_string, aliases, depth=0, max_depth=5):
    """Resolve a path string using defined aliases, handling nested aliases recursively."""
    if not path_string:
        return path_string
    if depth > max_depth:
        print(
            f"Warning: Max alias resolution depth ({max_depth}) exceeded for initial '{path_string}'. Returning unresolved."
        )
        # Return original string that caused the depth issue, not intermediate one
        return path_string

    resolved = path_string  # Start with the original string for this level

    # 1. Check for exact match alias (full path replacement)
    if resolved in aliases:
        # Recursively resolve the alias's value. Pass the ORIGINAL path_string for error reporting.
        resolved = resolve_path(aliases[resolved], aliases, depth + 1, max_depth)
        # If recursion hit max depth, return the result from that level
        if depth + 1 > max_depth and resolved == aliases[resolved]:
            return resolved  # Return the unresolved alias value causing the issue
        # Continue resolution in case the resolved value itself contains a prefix alias

    # 2. Check for prefix alias on the potentially updated 'resolved' string
    if ":" in resolved:
        alias_key, rest_of_path = resolved.split(":", 1)
        if alias_key in aliases:
            # Recursively resolve the base path alias first. Pass the ORIGINAL path_string.
            base_path_alias_value = aliases[alias_key]
            base_path = resolve_path(
                base_path_alias_value, aliases, depth + 1, max_depth
            )
            # If recursion hit max depth, return the result from that level
            if depth + 1 > max_depth and base_path == base_path_alias_value:
                return resolved  # Return the string with the unresolved prefix alias causing the issue

            # Concatenate carefully
            if base_path and rest_of_path:
                # Avoid double slash if base_path ends with / or rest_of_path starts with /
                if base_path.endswith("/") and rest_of_path.startswith("/"):
                    resolved = base_path + rest_of_path[1:]
                elif not base_path.endswith("/") and not rest_of_path.startswith("/"):
                    resolved = f"{base_path}/{rest_of_path}"
                else:  # One ends with / or the other starts with /
                    resolved = base_path + rest_of_path
            elif base_path:
                resolved = base_path  # Only base path resolved
            # If prefix resolution happened, we assume it's the final form for this level
            # We don't try to re-resolve the combined path in the same call
            return resolved

    # Return the final resolved path after potentially multiple steps at this level
    return resolved
