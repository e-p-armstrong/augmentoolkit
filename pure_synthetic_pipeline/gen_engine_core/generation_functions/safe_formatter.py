import string


class SafeFormatter(string.Formatter):
    def get_value(self, key, args, kwargs):
        if isinstance(key, str):
            return kwargs.get(key, "{" + key + "}")
        else:
            return super().get_value(key, args, kwargs)


def safe_format(format_string, *args, **kwargs):
    formatter = SafeFormatter()
    return formatter.format(format_string, *args, **kwargs)
