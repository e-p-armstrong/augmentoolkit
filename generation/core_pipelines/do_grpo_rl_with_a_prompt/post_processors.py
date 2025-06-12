REWARD_SCALING_FUNCTIONS = {}

REWARD_COMBINATION_FUNCTIONS = {}


def register_combination_function(name):
    def decorator(func):
        REWARD_COMBINATION_FUNCTIONS[name] = func
        return func

    return decorator


def register_scaling_function(name):
    def decorator(func):
        REWARD_SCALING_FUNCTIONS[name] = func
        return func

    return decorator


@register_scaling_function("identity")
def identity_scaling_function(x):
    return x


@register_combination_function("sum")
def sum_combination_function(x):
    return sum(x)


@register_combination_function("mean")
def mean_combination_function(x):
    return sum(x) / len(x)


def get_scaling_function_from_name(name):
    return REWARD_SCALING_FUNCTIONS[name]


def get_combination_function_from_name(name):
    return REWARD_COMBINATION_FUNCTIONS[name]
