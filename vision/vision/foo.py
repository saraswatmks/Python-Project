
def memoize(func):

    cache = {}

    def wrapper(*args, **kwargs):

        if kwargs not in cache:
            print(f'this is args: {args}')
            print(f'this is args type: {type(args)}')
            print(f"caching new value for {func.__name__}{args}")
            cache[kwargs] = func(*args, *kwargs)
        else:
            print(f"using old value for {func.__name__}{args}")

        return cache[kwargs]

    return wrapper

@memoize
def add(a, b):
    return a + b


if __name__ == '__main__':

    print(add(a=2, b=3))
    print(add(a=2, b=4))
    print(add(a=2, b=3))

