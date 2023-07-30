from nbag.gen_wrappers import wrap_module

if __name__ == "__main__":
    wrap_module("sympy", ".")
    wrap_module("sympy.stats", ".")
