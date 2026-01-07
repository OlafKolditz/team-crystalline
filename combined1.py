import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_I_D(a, b, phi_D, psi_D):
    """
    Calculate the dimensionless parameter I_D.
    
    Parameters:
    -----------
    a : float or array
        Characteristic length parameter (well spacing component)
    b : float or array
        Characteristic length parameter (aquifer thickness)
    phi_D : float or array
        Dimensionless hyperbolic coordinate parameter
    psi_D : float or array
        Dimensionless trigonometric coordinate parameter
    
    Returns:
    --------
    I_D : float or array
        Dimensionless injection/interaction parameter
        
    Formula:
    I_D = (2π/3)*(a²/b²)*[1 - (sinh(φ_D)/(cosh(φ_D)+cosh(ψ_D))) * 
                          (1 + cos(ψ_D)/(cos(φ_D)+cos(ψ_D)))]
    """
    # Calculate geometric prefactor
    geometric_factor = (2 * np.pi / 3) * (a**2 / b**2)
    
    # Calculate hyperbolic terms
    sinh_phi = np.sinh(phi_D)
    cosh_phi = np.cosh(phi_D)
    cosh_psi = np.cosh(psi_D)
    
    # Calculate trigonometric terms
    cos_phi = np.cos(phi_D)
    cos_psi = np.cos(psi_D)
    
    # Calculate the two main terms
    # Term1: sinh(φ_D)/(cosh(φ_D)+cosh(ψ_D))
    term1 = sinh_phi / (cosh_phi + cosh_psi)
    
    # Term2: 1 + cos(ψ_D)/(cos(φ_D)+cos(ψ_D))
    term2 = 1 + cos_psi / (cos_phi + cos_psi)
    
    # Complete expression inside braces
    inside_braces = 1 - term1 * term2
    
    # Final result
    I_D = geometric_factor * inside_braces
    
    return I_D

def calculate_T_D(t_D, I_D, alpha, beta):
    """
    Calculate dimensionless temperature T_D.
    
    Parameters:
    -----------
    t_D : float or array
        Dimensionless time
    I_D : float or array
        Dimensionless parameter from calculate_I_D()
    alpha : float
        Thermal diffusivity parameter
    beta : float
        Convection/conductance parameter
        
    Returns:
    --------
    T_D : float or array
        Dimensionless temperature
        
    Formula:
    T_D = u(t_D - I_D) * erfc(β*I_D / √[α*(t_D - I_D)])
    where u() is the Heaviside step function
    """
    # Ensure inputs are arrays for consistent handling
    t_D = np.asarray(t_D)
    I_D = np.asarray(I_D) if not np.isscalar(I_D) else I_D
    
    # Initialize result array
    T_D = np.zeros_like(t_D, dtype=float)
    
    # Apply Heaviside step function: u(t_D - I_D) = 0 when t_D <= I_D, 1 when t_D > I_D
    mask = t_D > I_D
    
    if np.any(mask):
        # Calculate the argument for erfc
        # For array I_D with scalar t_D or vice versa, need to handle broadcasting
        if np.isscalar(I_D):
            I_D_val = I_D
        else:
            I_D_val = I_D[mask]
        
        argument = beta * I_D_val / np.sqrt(alpha * (t_D[mask] - I_D_val))
        T_D[mask] = erfc(argument)
    
    return T_D

def combined_solution(t_D, a, b, phi_D, psi_D, alpha, beta):
    """
    Combined solution: calculate I_D first, then T_D.
    
    Parameters:
    -----------
    t_D : float or array
        Dimensionless time
    a, b, phi_D, psi_D : float or array
        Parameters for I_D calculation
    alpha, beta : float
        Parameters for T_D calculation
        
    Returns:
    --------
    I_D : float or array
        Dimensionless parameter
    T_D : float or array
        Dimensionless temperature
    """
    # Calculate I_D
    I_D_val = calculate_I_D(a, b, phi_D, psi_D)
    
    # Calculate T_D using I_D
    T_D_val = calculate_T_D(t_D, I_D_val, alpha, beta)
    
    return I_D_val, T_D_val

def plot_breakthrough_curves(t_D_range=(0, 20), a=1.0, b=0.5, phi_D=1.0, 
                             psi_D=0.5, alpha=1.0, beta_values=[0.5, 1.0, 2.0, 4.0]):
    """
    Plot temperature breakthrough curves for different beta values.
    
    Parameters:
    -----------
    t_D_range : tuple
        Range of dimensionless time (start, end)
    a, b, phi_D, psi_D : float
        Parameters for I_D calculation
    alpha : float
        Thermal diffusivity parameter
    beta_values : list
        List of beta parameter values to compare
    """
    # Create time array
    t_D = np.linspace(t_D_range[0], t_D_range[1], 1000)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # Calculate I_D once (constant for fixed parameters)
    I_D_val = calculate_I_D(a, b, phi_D, psi_D)
    
    # Plot 1: Temperature breakthrough for different beta
    ax1 = axes[0, 0]
    for beta in beta_values:
        T_D = calculate_T_D(t_D, I_D_val, alpha, beta)
        ax1.plot(t_D, T_D, label=f'β = {beta}', linewidth=2)
    
    ax1.axvline(x=I_D_val, color='red', linestyle='--', alpha=0.7, 
                label=f'I_D = {I_D_val:.3f}')
    ax1.set_xlabel('Dimensionless Time (t_D)', fontsize=12)
    ax1.set_ylabel('Dimensionless Temperature (T_D)', fontsize=12)
    ax1.set_title(f'Temperature Breakthrough Curves\n(I_D = {I_D_val:.3f}, α = {alpha})', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xlim(t_D_range)
    ax1.set_ylim(-0.05, 1.05)
    
    # Plot 2: Effect of I_D on breakthrough (fixed beta)
    ax2 = axes[0, 1]
    beta_fixed = 1.0
    a_values = [0.5, 1.0, 1.5, 2.0]
    
    for a_val in a_values:
        I_D_temp = calculate_I_D(a_val, b, phi_D, psi_D)
        T_D_temp = calculate_T_D(t_D, I_D_temp, alpha, beta_fixed)
        ax2.plot(t_D, T_D_temp, label=f'a={a_val}, I_D={I_D_temp:.3f}', linewidth=2)
    
    ax2.set_xlabel('Dimensionless Time (t_D)', fontsize=12)
    ax2.set_ylabel('Dimensionless Temperature (T_D)', fontsize=12)
    ax2.set_title(f'Effect of Well Spacing (a) on Breakthrough\n(β = {beta_fixed}, α = {alpha})', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='lower right')
    ax2.set_xlim(t_D_range)
    ax2.set_ylim(-0.05, 1.05)
    
    # Plot 3: Effect of alpha on breakthrough
    ax3 = axes[1, 0]
    beta_fixed = 1.0
    alpha_values = [0.5, 1.0, 2.0, 4.0]
    
    for alpha_val in alpha_values:
        T_D_temp = calculate_T_D(t_D, I_D_val, alpha_val, beta_fixed)
        ax3.plot(t_D, T_D_temp, label=f'α = {alpha_val}', linewidth=2)
    
    ax3.axvline(x=I_D_val, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Dimensionless Time (t_D)', fontsize=12)
    ax3.set_ylabel('Dimensionless Temperature (T_D)', fontsize=12)
    ax3.set_title(f'Effect of Thermal Diffusivity (α)\n(β = {beta_fixed}, I_D = {I_D_val:.3f})', fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_xlim(t_D_range)
    ax3.set_ylim(-0.05, 1.05)
    
    # Plot 4: I_D as function of phi_D and psi_D
    ax4 = axes[1, 1]
    phi_vals = np.linspace(-2, 2, 100)
    psi_vals = np.linspace(-2, 2, 100)
    Phi, Psi = np.meshgrid(phi_vals, psi_vals)
    
    I_D_grid = calculate_I_D(a, b, Phi, Psi)
    contour = ax4.contourf(Phi, Psi, I_D_grid, levels=50, cmap='viridis')
    
    # Mark the point used in other plots
    ax4.plot(phi_D, psi_D, 'ro', markersize=10, 
             label=f'Current: φ={phi_D}, ψ={psi_D}')
    
    ax4.set_xlabel(r'$\phi_D$', fontsize=12)
    ax4.set_ylabel(r'$\psi_D$', fontsize=12)
    ax4.set_title(r'$I_D(\phi_D, \psi_D)$ Contour', fontsize=13)
    ax4.legend(fontsize=10)
    fig.colorbar(contour, ax=ax4, label='I_D')
    
    # Add parameter info text
    param_text = f'Fixed parameters:\n'
    param_text += f'a = {a}, b = {b}\n'
    param_text += f'φ_D = {phi_D}, ψ_D = {psi_D}\n'
    param_text += f'a/b = {a/b:.2f}\n'
    param_text += f'I_D = {I_D_val:.3f}'
    
    plt.figtext(0.02, 0.02, param_text, fontsize=10, 
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Combined Solution: Geothermal Heat Extraction Model', 
                 fontsize=15, y=0.98)
    plt.tight_layout()
    plt.show()
    
    return t_D, I_D_val

def analyze_thermal_breakthrough():
    """
    Detailed analysis of thermal breakthrough time.
    """
    print("=" * 70)
    print("THERMAL BREAKTHROUGH ANALYSIS")
    print("=" * 70)
    
    # Define parameters
    a, b = 1.0, 0.5
    phi_D, psi_D = 1.0, 0.5
    alpha, beta = 1.0, 1.0
    
    # Calculate I_D
    I_D_val = calculate_I_D(a, b, phi_D, psi_D)
    print(f"\n1. Dimensionless Parameter I_D:")
    print(f"   a = {a}, b = {b}, φ_D = {phi_D}, ψ_D = {psi_D}")
    print(f"   I_D = {I_D_val:.6f}")
    
    # Calculate T_D at various times
    print(f"\n2. Temperature Evolution (α = {alpha}, β = {beta}):")
    print("-" * 60)
    print(f"{'t_D':>8} {'t_D - I_D':>12} {'Argument':>15} {'T_D':>12}")
    print("-" * 60)
    
    t_D_vals = np.array([0.5, 1.0, I_D_val, I_D_val+0.1, I_D_val+0.5, I_D_val+1.0, I_D_val+2.0])
    
    for t in t_D_vals:
        T_D_val = calculate_T_D(t, I_D_val, alpha, beta)
        if t <= I_D_val:
            arg = np.nan
        else:
            arg = beta * I_D_val / np.sqrt(alpha * (t - I_D_val))
        
        print(f"{t:8.3f} {t-I_D_val:12.6f} ", end="")
        if t <= I_D_val:
            print(f"{'---':>15} {T_D_val:12.6f}")
        else:
            print(f"{arg:15.6f} {T_D_val:12.6f}")
    
    print("-" * 60)
    
    # Calculate time to reach specific temperatures
    print(f"\n3. Time to Reach Specific Temperatures:")
    
    # Define target temperatures
    target_T = [0.1, 0.25, 0.5, 0.75, 0.9]
    
    # Numerical solution for t_D given T_D
    def find_t_D_for_T(target_T_D, I_D_val, alpha, beta, t_min=I_D_val+0.001, t_max=I_D_val+100):
        """Find t_D that gives specific T_D value."""
        # Define function to find root
        def f(t):
            return calculate_T_D(t, I_D_val, alpha, beta) - target_T_D
        
        # Use bisection method
        t_low, t_high = t_min, t_max
        f_low, f_high = f(t_low), f(t_high)
        
        if f_low * f_high > 0:
            return None  # No solution in range
        
        for _ in range(50):  # Max iterations
            t_mid = (t_low + t_high) / 2
            f_mid = f(t_mid)
            
            if abs(f_mid) < 1e-6:
                return t_mid
            
            if f_low * f_mid < 0:
                t_high = t_mid
                f_high = f_mid
            else:
                t_low = t_mid
                f_low = f_mid
        
        return (t_low + t_high) / 2
    
    print("-" * 40)
    print(f"{'Target T_D':>12} {'Required t_D':>15} {'Delay (t_D - I_D)':>18}")
    print("-" * 40)
    
    for T_target in target_T:
        t_required = find_t_D_for_T(T_target, I_D_val, alpha, beta)
        if t_required:
            print(f"{T_target:12.2f} {t_required:15.6f} {t_required - I_D_val:18.6f}")
        else:
            print(f"{T_target:12.2f} {'Not reached':>15} {'---':>18}")
    
    print("-" * 40)
    
    return I_D_val

# Example usage and demonstration
if __name__ == "__main__":
    print("=" * 70)
    print("COMBINED SOLUTION FOR GEOTHERMAL HEAT EXTRACTION")
    print("=" * 70)
    
    # Display the formulas
    print("\nFormulas implemented:")
    print("1. I_D = (2π/3)*(a²/b²)*[1 - A * (1 + B)]")
    print("   where A = sinh(φ_D)/(cosh(φ_D)+cosh(ψ_D))")
    print("         B = cos(ψ_D)/(cos(φ_D)+cos(ψ_D))")
    print("\n2. T_D = u(t_D - I_D) * erfc(β*I_D / √[α*(t_D - I_D)])")
    print()
    
    # Example 1: Single point calculation
    print("Example 1: Single point calculation")
    t_D_single = 5.0
    a_val, b_val = 1.0, 0.5
    phi_val, psi_val = 1.0, 0.5
    alpha_val, beta_val = 0.8, 1.2
    
    I_D_result, T_D_result = combined_solution(t_D_single, a_val, b_val, 
                                               phi_val, psi_val, 
                                               alpha_val, beta_val)
    
    print(f"Parameters:")
    print(f"  t_D = {t_D_single}")
    print(f"  a = {a_val}, b = {b_val}, φ_D = {phi_val}, ψ_D = {psi_val}")
    print(f"  α = {alpha_val}, β = {beta_val}")
    print(f"\nResults:")
    print(f"  I_D = {I_D_result:.6f}")
    print(f"  T_D = {T_D_result:.6f}")
    
    # Example 2: Time series calculation
    print("\n" + "=" * 70)
    print("Example 2: Time series calculation")
    
    t_D_array = np.linspace(0, 10, 21)  # 0 to 10 in 21 points
    I_D_array, T_D_array = combined_solution(t_D_array, a_val, b_val, 
                                             phi_val, psi_val, 
                                             alpha_val, beta_val)
    
    print(f"\nTime evolution (a={a_val}, b={b_val}, α={alpha_val}, β={beta_val}):")
    print("-" * 65)
    print(f"{'t_D':>6} {'I_D':>10} {'t_D > I_D?':>12} {'T_D':>12}")
    print("-" * 65)
    
#    for t, I, T in zip(t_D_array, I_D_array, T_D_array):
#        comparison = "Yes" if t > I else "No"
#        print(f"{t:6.2f} {I:10.6f} {comparison:>12} {T:12.6f}")
    
    print("-" * 65)
    
    # Run detailed analysis
    print("\n" + "=" * 70)
    analyze_thermal_breakthrough()
    
    # Generate comprehensive plots
    print("\n" + "=" * 70)
    print("Generating comprehensive visualization...")
    
    # Plot breakthrough curves
    t_D_vals, I_D_fixed = plot_breakthrough_curves(
        t_D_range=(0, 15),
        a=1.0,
        b=0.5,
        phi_D=1.0,
        psi_D=0.5,
        alpha=1.0,
        beta_values=[0.3, 0.7, 1.5, 3.0]
    )
    
    # Additional analysis: Sensitivity to phi_D and psi_D
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS: Effect of φ_D and ψ_D on I_D")
    print("=" * 70)
    
    a_test, b_test = 1.0, 0.5
    phi_test_cases = [0, 0.5, 1.0, 1.5]
    psi_test_cases = [0, 0.5, 1.0, 1.5]
    
    print(f"\nI_D values for a={a_test}, b={b_test}:")
    print("-" * 50)
    print(f"{'φ_D':>8} {'ψ_D':>8} {'I_D':>12} {'a²/b²':>12} {'Inside []':>12}")
    print("-" * 50)
    
    for phi in phi_test_cases:
        for psi in psi_test_cases:
            I_D_test = calculate_I_D(a_test, b_test, phi, psi)
            a2_b2 = a_test**2 / b_test**2
            inside_val = I_D_test / ((2 * np.pi / 3) * a2_b2)
            print(f"{phi:8.2f} {psi:8.2f} {I_D_test:12.6f} {a2_b2:12.4f} {inside_val:12.6f}")
    
    print("-" * 50)