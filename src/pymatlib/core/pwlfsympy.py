import os
import pwlf
import numpy as np
import sympy as sp
from typing import Union
import matplotlib.pyplot as plt
import pymatlib.core.data_handler as dh
from pymatlib.core.symbol_registry import SymbolRegistry


seed = 13579


#https://github.com/cjekel/piecewise_linear_fit_py/blob/master/examples/understanding_higher_degrees/polynomials_in_pwlf.ipynb
def get_symbolic_eqn(pwlf_: pwlf.PiecewiseLinFit, segment_number: int, x: Union[float, sp.Symbol]):
    if pwlf_.degree < 1:
        raise ValueError('Degree must be at least 1')
    if segment_number < 1 or segment_number > pwlf_.n_segments:
        raise ValueError('segment_number not possible')
    # Check if x is a symbolic variable
    is_symbolic = isinstance(x, (sp.Symbol, sp.Expr))
    # assemble degree = 1 first
    for line in range(segment_number):
        if line == 0:
            my_eqn = pwlf_.beta[0] + (pwlf_.beta[1])*(x-pwlf_.fit_breaks[0])
        else:
            my_eqn += (pwlf_.beta[line+1])*(x-pwlf_.fit_breaks[line])
    # assemble all other degrees
    if pwlf_.degree > 1:
        for k in range(2, pwlf_.degree + 1):
            for line in range(segment_number):
                beta_index = pwlf_.n_segments*(k-1) + line + 1
                my_eqn += (pwlf_.beta[beta_index])*(x-pwlf_.fit_breaks[line])**k
    # Only call simplify if x is symbolic
    if is_symbolic:
        print(f'my_eqn.simplify(): {my_eqn.simplify()}')
        return my_eqn.simplify()
    else:
        # For numeric x, just return the equation
        return my_eqn

# https://github.com/cjekel/piecewise_linear_fit_py/blob/master/examples/understanding_higher_degrees/polynomials_in_pwlf.ipynb
def get_symbolic_conditions(pwlf_: pwlf.PiecewiseLinFit, x: sp.Symbol, lower_: str, upper_: str):
    conditions = []

    # Special case for 1 segment
    if pwlf_.n_segments == 1:
        eqn = get_symbolic_eqn(pwlf_, 1, x)

        # Handle lower boundary
        if lower_ == "extrapolate":
            # Single condition for 1 segment with extrapolation at both ends
            if upper_ == "extrapolate":
                conditions.append((eqn, True))
                return conditions
            # Otherwise, handle lower extrapolation
            conditions.append((eqn, sp.And(x < pwlf_.fit_breaks[1])))
        elif lower_ == "constant":
            # Add constant value for lower boundary
            conditions.append((eqn.evalf(subs={x: pwlf_.fit_breaks[0]}), x < pwlf_.fit_breaks[0]))
            # Add condition for the segment
            conditions.append((eqn, sp.And(x >= pwlf_.fit_breaks[0], x < pwlf_.fit_breaks[1])))

        # Handle upper boundary
        if upper_ == "extrapolate":
            # Add extrapolation for upper boundary
            conditions.append((eqn, x >= pwlf_.fit_breaks[1]))
        elif upper_ == "constant":
            # Add constant value for upper boundary
            conditions.append((eqn.evalf(subs={x: pwlf_.fit_breaks[1]}), x >= pwlf_.fit_breaks[1]))

        return conditions

    for i in range(pwlf_.n_segments):
        eqn = get_symbolic_eqn(pwlf_, i + 1, x)
        # print('Equation number: ', i + 1)
        # print(eqn_list[-1])
        # f_list.append(sp.lambdify(T, eqn_list[-1]))
        if i == 0:
            if lower_ == "extrapolate":
                conditions.append((eqn, sp.And(x < pwlf_.fit_breaks[i+1])))
            elif lower_ == "constant":
                conditions.append((eqn.evalf(subs={x: pwlf_.fit_breaks[i]}), sp.And(x < pwlf_.fit_breaks[i])))
                conditions.append((eqn, sp.And(x >= pwlf_.fit_breaks[i], x < pwlf_.fit_breaks[i + 1])))
        elif i == pwlf_.n_segments - 1:
            if upper_ == "extrapolate":
                conditions.append((eqn, True))
            elif upper_ == "constant":
                conditions.append((eqn, sp.And(x >= pwlf_.fit_breaks[i], x < pwlf_.fit_breaks[i + 1])))
                conditions.append((eqn.evalf(subs={x: pwlf_.fit_breaks[i + 1]}), True))
        else:
            conditions.append((eqn, sp.And(x >= pwlf_.fit_breaks[i], x < pwlf_.fit_breaks[i + 1])))
    print(f'conditions: {conditions}')
    return conditions


def create_pwlf(name: str, x: np.ndarray, y: np.ndarray, v_deg=1, v_seg=3, lower='constant', upper='constant', show=True):
    v_pwlf = pwlf.PiecewiseLinFit(x, y, degree=v_deg, seed=seed)
    print(f'v_pwlf: {v_pwlf}')
    # other fit functions are possible e.g. with fixed break points
    v_pwlf.fit(v_seg)
    print(f'v_pwlf.fit(v_seg): {v_pwlf.fit(v_seg)}')
    pw = sp.Piecewise(*get_symbolic_conditions(v_pwlf, T, lower, upper))
    print(f'pw: {pw}')
    if show:
        plt.figure()
        plt.title(name)
        plt.plot(x, y, linewidth=1, marker='o', markersize=1., label='measurement')
        plt.plot(x, v_pwlf.predict(x), linestyle='-', linewidth=1, label='pwlf')
        f = sp.lambdify(T, pw, 'numpy')  # returns a numpy-ready function
        plt.plot(x, f(x), linestyle=':', linewidth=1, label='symb')
        plt.legend()

        # Define filename and directory
        filename = f"{name.replace(' ', '_').replace('/', '_')}.png"
        directory = "pwlf_plots"
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists

        filepath = os.path.join(directory, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        # plt.show()
        print(f"Plot saved as {filepath}")
    return pw


def read_data(file_path: str, x: str, y: str, v_deg=1, v_seg=3, lower='constant', upper='constant') -> sp.Piecewise:
    T_xls, v_xls = dh.read_data_from_excel(file_path, x, y)
    if T_xls[0] > T_xls[-1]:
        T_xls = np.flip(T_xls)
        v_xls = np.flip(v_xls)
    pw = create_pwlf(y, T_xls, v_xls, v_deg=v_deg, v_seg=v_seg, lower=lower, upper=upper)
    return pw

"""
def read_data1(file_path: str, x: str, y: str, v_deg=1, v_seg=3, lower='constant', upper='constant', show=True):
    #
    # This data is now hard coded, but needs to be read from yaml file
    #
    T_xls, v_xls = dh.read_data_from_excel(file_path, x, y)

    if T_xls[0] > T_xls[-1]:
        T_xls = np.flip(T_xls)
        v_xls = np.flip(v_xls)
    v_pwlf = pwlf.PiecewiseLinFit(T_xls, v_xls, degree=v_deg, seed=seed)
    print(f'v_pwlf: {v_pwlf}')
    # other fit functions are possible e.g. with fixed break points
    v_pwlf.fit(v_seg)
    print(f'v_pwlf.fit(v_seg): {v_pwlf.fit(v_seg)}')
    pw = sp.Piecewise(*get_symbolic_conditions(v_pwlf, T, lower, upper))
    print(f'pw: {pw}')

    if show:
        plt.figure()
        plt.title(y)
        plt.plot(T_xls, v_xls, linewidth=1, marker='o', markersize=2, label='measurement')
        plt.plot(T_xls, v_pwlf.predict(T_xls), linestyle='-', linewidth=1, label='pwlf')
        f = sp.lambdify(T, pw, 'numpy')  # returns a numpy-ready function
        plt.plot(T_xls, f(T_xls), linestyle=':', linewidth=1, label='symb')
        plt.legend()

        # Define filename and directory
        filename = f"{y.replace(' ', '_').replace('/', '_')}.png"
        directory = "pwlf_plots"
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists

        filepath = os.path.join(directory, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.show()
        print(f"Plot saved as {filepath}")

    return pw
"""

def show_symb(symbol, x, expr, title):
    plt.title(title)
    f = sp.lambdify(symbol, expr, 'numpy')  # returns a numpy-ready function
    print(f'f: {f}')
    plt.plot(x, f(x), ':', label='symb')
    plt.legend()
    plt.show()


def solve(variables_):
    print("solve")
    print(f"variables_: {variables_}")
    # missing is a cycle detection
    found = True
    while found:
        found = False
        simplified = dict()
        for key in variables_:
            print(f"key: {key}")
            eq = variables_[key]
            print(f"eq: {eq}")
            print(f"type(eq): {type(eq)}")
            if eq.has_free:
                for free_symbol in variables_[key].free_symbols:
                    if free_symbol in variables_:
                        print(f"free_symbol: {free_symbol}")
                        print(f"variables_[free_symbol]: {variables_[free_symbol]}")
                        eq = eq.subs([(free_symbol, variables_[free_symbol])])
                        print(f"eq: {eq}")
                        found = True
                simplified[key] = eq
        variables_ = simplified
        print(f"variables_: {variables_}")
    for key in variables_:
        variables_[key] = variables_[key].doit()
        print(f"variables_[key]: {variables_[key]}")
    print(f'variables_: {variables_}')
    print(f"type(variables_): {type(variables_)}")
    return variables_


def approximate(symbol: sp.Symbol, x: np.ndarray, expr: sp.Expr, v_deg=1, v_seg=3, lower='constant', upper='constant', show=True, title : str="") -> sp.Piecewise:
    f = sp.lambdify(symbol, expr, 'numpy')  # returns a numpy-ready function
    v_pwlf = pwlf.PiecewiseLinFit(x, f(x), degree=v_deg, seed=seed)
    v_pwlf.fit(v_seg)
    pw = sp.Piecewise(*get_symbolic_conditions(v_pwlf, T, lower, upper))
    print(f'approx pw for {title}: ', pw)

    if show:
        plt.figure()
        plt.title(f"approx_{str(title)}")
        plt.plot(x, f(x), '-', label='func')
        g = sp.lambdify(T, pw, 'numpy')  # returns a numpy-ready function
        plt.plot(x, g(x), ':', label='approx')
        plt.legend()

        # Define filename and directory
        filename = f"approx_{title}.png"
        directory = "pwlf_plots"
        os.makedirs(directory, exist_ok=True)  # Ensure the directory exists

        filepath = os.path.join(directory, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        # plt.show()
        print(f"Plot saved as {filepath}")
    return pw


if __name__ == '__main__':
    T = sp.S("T")

    file_path = '../data/alloys/SS304L/304L_Erstarrungsdaten_edited.xlsx'

    variables = dict()
    print(f"variables: {variables}")

    s_temperature_liquidus = SymbolRegistry.get("liquidus_temperature")
    print(f's_temperature_liquidus: {s_temperature_liquidus}')
    print(f"type(s_temperature_liquidus): {type(s_temperature_liquidus)}")
    variables[s_temperature_liquidus] = sp.Float(1720.)
    print(f"variables[s_temperature_liquidus]: {variables.get(s_temperature_liquidus)}")
    print(f"type(variables[s_temperature_liquidus]): {type(variables.get(s_temperature_liquidus))}")
    print(f"variables: {variables}")

    s_thermal_expansion_coefficient = SymbolRegistry.get("thermal_expansion_coefficient")
    variables[s_thermal_expansion_coefficient] = sp.Float("16.3e-6")
    print(f"type(variables[s_thermal_expansion_coefficient]): {type(variables[s_thermal_expansion_coefficient])}")

    #
    # This data is now hard coded, but needs to be read from yaml file
    #
        #'Thermal conductivity (W/(m*K))-TOTAL-10000.0(K/s)')
        #'Liquid viscosity (mPa s)-  -10000.0(K/s)')

    s_density = SymbolRegistry.get("density")
    # variables[s_density] = sp.Float(7890)
    variables[s_density] = read_data(file_path, 'T (K)', 'Density (kg/(m)^3)', v_deg=1, v_seg=3, lower='extrapolate', upper='extrapolate')
    # variables[s_density] = create_pwlf('create_pwlf_density', np.array([1, 2, 3, 4], dtype=np.float64), np.array([1, 4, 9, 16], dtype=np.float64), v_deg=2, v_seg=4, lower='extrapolate', upper='extrapolate')
    # TODO!
    # variables[s_density] = create_pwlf('density_create_pwlf', np.array([1, 2, 3, 4], dtype=np.float64), np.array([T, T+1, T+2], dtype=np.float64), v_deg=2, v_seg=4, lower='extrapolate', upper='extrapolate')

    print(f"variables[s_density]: {variables.get(s_density)}")
    print(f"type(variables[s_density]): {type(variables.get(s_density))}")
    # print(f"variables: {variables}")

    s_heat_capacity = SymbolRegistry.get("heat_capacity")
    variables[s_heat_capacity] = read_data(file_path, 'T (K)', 'Specific heat (J/(Kg K))', v_deg=1, v_seg=4, lower='constant', upper='constant')

    s_heat_conductivity = SymbolRegistry.get("heat_conductivity")
    variables[s_heat_conductivity] = read_data(file_path, 'T (K)', 'Thermal conductivity (W/(m*K))-TOTAL-10000.0(K/s)', v_deg=1, v_seg=4, lower='extrapolate', upper='extrapolate')

    s_thermal_diffusivity = SymbolRegistry.get("thermal_diffusivity")
    variables[s_thermal_diffusivity] = sp.sympify("heat_conductivity / (density * heat_capacity)")
    print(f"thermal_diffusivity: {variables[s_thermal_diffusivity]}")
    print(f"type(variables[s_thermal_diffusivity]): {type(variables[s_thermal_diffusivity])}")

    s_enthalpy = SymbolRegistry.get("enthalpy")
    variables[s_enthalpy] = sp.sympify("Integral(heat_capacity, T)", evaluate=False)
    print(f"variables[s_enthalpy]: {variables[s_enthalpy]}")
    print(f"type(variables[s_enthalpy]): {type(variables[s_enthalpy])}")
    print(f"variables: {variables}")

    print("#-#" * 100)
    variables = solve(variables)
    print("#-#" * 100)
    print(f"variables[s_density]: {variables[s_density]}")
    print(f"variables[s_thermal_diffusivity]: {variables[s_thermal_diffusivity]}")
    print(f"variables[s_enthalpy]: {variables.get(s_enthalpy)}")
    print(f"variables: {variables}, type(variables): {type(variables)}")
    print("Variables debug information:")
    print("-" * 100)
    for var_name, var_value in variables.items():
        print(f"Variable: {var_name}, Type: {type(var_name).__name__}")
        print(f"Value:\n{var_value}")
        print(f"Value type: {type(var_value)}")
        print("-" * 100)
    print(SymbolRegistry.get_all())
    #sp.pprint(variables[s_enthalpy])
    #sp.pprint(variables[s_heat_capacity])
    #sp.pprint(variables[s_heat_conductivity])
    #sp.pprint(variables[s_thermal_diffusivity])
    #sp.pprint(sp.simplify(variables[s_thermal_diffusivity]))

    T_data = np.linspace(300, 3000, 541)
    # print(f"T_data: {T_data}")

    #show_symb(T, T_data, variables[s_enthalpy], s_enthalpy)
    #show_symb(T, T_data, variables[s_thermal_diffusivity], s_thermal_diffusivity)

    variables[s_density] = approximate(T, T_data, variables[s_density], v_deg=1, v_seg=3, lower='constant', upper='constant', title=s_density)
    variables[s_thermal_diffusivity] = approximate(T, T_data, variables[s_thermal_diffusivity], v_deg=1, v_seg=3, lower='extrapolate', upper='extrapolate', title=s_thermal_diffusivity)
    variables[s_enthalpy] = approximate(T, T_data, variables[s_enthalpy], v_deg=2, v_seg=3, lower='extrapolate', upper='extrapolate', title=s_enthalpy)
