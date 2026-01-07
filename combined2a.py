import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def phi_D(x_D, y_D):
    """
    Calculate the dimensionless hyperbolic coordinate φ_D.
    
    Parameters:
    -----------
    x_D : float or array
        Dimensionless x-coordinate (x/L, where L is characteristic length)
    y_D : float or array
        Dimensionless y-coordinate (y/L)
    
    Returns:
    --------
    phi_D_val : float or array
        Dimensionless hyperbolic coordinate
        
    Formula:
    φ_D = 0.5 * ln[((x_D+1)² + y_D²) / ((x_D-1)² + y_D²)]
    """
    numerator = (x_D + 1)**2 + y_D**2
    denominator = (x_D - 1)**2 + y_D**2
    
    # Handle case where denominator is 0 or negative
    denominator = np.where(denominator <= 0, 1e-10, denominator)
    
    return 0.5 * np.log(numerator / denominator)

def psi_D(x_D, y_D):
    """
    Calculate the dimensionless trigonometric coordinate ψ_D.
    
    Parameters:
    -----------
    x_D : float or array
        Dimensionless x-coordinate
    y_D : float or array
        Dimensionless y-coordinate
    
    Returns:
    --------
    psi_D_val : float or array
        Dimensionless trigonometric coordinate (in radians)
        
    Formula:
    ψ_D = arctan(2*y_D / (1 - x_D² - y_D²))
    
    Note: Uses np.arctan2 for correct quadrant handling
    """
    numerator = 2 * y_D
    denominator = 1 - x_D**2 - y_D**2
    
    # Use arctan2 for proper quadrant handling
    return np.arctan2(numerator, denominator)

def calculate_I_D(a, b, phi_D_val, psi_D_val):
    """
    Calculate the dimensionless parameter I_D.
    
    Parameters:
    -----------
    a : float or array
        Characteristic length parameter (well spacing component)
    b : float or array
        Characteristic length parameter (aquifer thickness)
    phi_D_val : float or array
        Dimensionless hyperbolic coordinate from phi_D()
    psi_D_val : float or array
        Dimensionless trigonometric coordinate from psi_D()
    
    Returns:
    --------
    I_D_val : float or array
        Dimensionless injection/interaction parameter
        
    Formula:
    I_D = (2π/3)*(a²/b²)*[1 - (sinh(φ_D)/(cosh(φ_D)+cosh(ψ_D))) * 
                          (1 + cos(ψ_D)/(cos(φ_D)+cos(ψ_D)))]
    """
    # Calculate geometric prefactor
    geometric_factor = (2 * np.pi / 3) * (a**2 / b**2)
    
    # Calculate hyperbolic terms
    sinh_phi = np.sinh(phi_D_val)
    cosh_phi = np.cosh(phi_D_val)
    cosh_psi = np.cosh(psi_D_val)
    
    # Calculate trigonometric terms
    cos_phi = np.cos(phi_D_val)
    cos_psi = np.cos(psi_D_val)
    
    # Calculate the two main terms
    # Term1: sinh(φ_D)/(cosh(φ_D)+cosh(ψ_D))
    term1 = sinh_phi / (cosh_phi + cosh_psi)
    
    # Term2: 1 + cos(ψ_D)/(cos(φ_D)+cos(ψ_D))
    term2 = 1 + cos_psi / (cos_phi + cos_psi)
    
    # Complete expression inside braces
    inside_braces = 1 - term1 * term2
    
    # Final result
    I_D_val = geometric_factor * inside_braces
    
    return I_D_val

def calculate_T_D(t_D, I_D_val, alpha, beta):
    """
    Calculate dimensionless temperature T_D.
    
    Parameters:
    -----------
    t_D : float or array
        Dimensionless time
    I_D_val : float or array
        Dimensionless parameter from calculate_I_D()
    alpha : float
        Thermal diffusivity parameter
    beta : float
        Convection/conductance parameter
        
    Returns:
    --------
    T_D_val : float or array
        Dimensionless temperature
        
    Formula:
    T_D = u(t_D - I_D) * erfc(β*I_D / √[α*(t_D - I_D)])
    where u() is the Heaviside step function
    """
    # Ensure inputs are arrays for consistent handling
    t_D = np.asarray(t_D)
    I_D_val = np.asarray(I_D_val) if not np.isscalar(I_D_val) else I_D_val
    
    # Initialize result array
    T_D_val = np.zeros_like(t_D, dtype=float)
    
    # Apply Heaviside step function: u(t_D - I_D) = 0 when t_D <= I_D, 1 when t_D > I_D
    mask = t_D > I_D_val
    
    if np.any(mask):
        # Calculate the argument for erfc
        # For array I_D with scalar t_D or vice versa, need to handle broadcasting
        if np.isscalar(I_D_val):
            I_D_arg = I_D_val
        else:
            I_D_arg = I_D_val[mask]
        
        argument = beta * I_D_arg / np.sqrt(alpha * (t_D[mask] - I_D_arg))
        T_D_val[mask] = erfc(argument)
    
    return T_D_val

def complete_solution(t_D, x_D, y_D, a, b, alpha, beta):
    """
    Complete solution: calculate φ_D, ψ_D, then I_D, then T_D.
    
    Parameters:
    -----------
    t_D : float or array
        Dimensionless time
    x_D, y_D : float or array
        Dimensionless coordinates
    a, b : float
        Geometric parameters for I_D calculation
    alpha, beta : float
        Thermal parameters for T_D calculation
        
    Returns:
    --------
    phi_D_val : float or array
        Hyperbolic coordinate
    psi_D_val : float or array
        Trigonometric coordinate
    I_D_val : float or array
        Dimensionless parameter
    T_D_val : float or array
        Dimensionless temperature
    """
    # Step 1: Calculate φ_D and ψ_D from x_D and y_D
    phi_D_val = phi_D(x_D, y_D)
    psi_D_val = psi_D(x_D, y_D)
    
    # Step 2: Calculate I_D from φ_D, ψ_D, a, b
    I_D_val = calculate_I_D(a, b, phi_D_val, psi_D_val)
    
    # Step 3: Calculate T_D from t_D, I_D, α, β
    T_D_val = calculate_T_D(t_D, I_D_val, alpha, beta)
    
    return phi_D_val, psi_D_val, I_D_val, T_D_val

def plot_coordinate_transformation(x_range=(-3, 3), y_range=(-3, 3)):
    """
    Visualize the transformation from (x_D, y_D) to (φ_D, ψ_D).
    """
    # Create grid in physical coordinates
    x_vals = np.linspace(x_range[0], x_range[1], 200)
    y_vals = np.linspace(y_range[0], y_range[1], 200)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Calculate φ_D and ψ_D
    Phi = phi_D(X, Y)
    Psi = psi_D(X, Y)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: φ_D as function of x_D, y_D
    ax1 = axes[0, 0]
    contour1 = ax1.contourf(X, Y, Phi, levels=50, cmap='RdBu_r')
    ax1.set_xlabel(r'$x_D$', fontsize=12)
    ax1.set_ylabel(r'$y_D$', fontsize=12)
    ax1.set_title(r'$\phi_D(x_D, y_D)$', fontsize=13)
    ax1.set_aspect('equal')
    fig.colorbar(contour1, ax=ax1, label=r'$\phi_D$')
    
    # Mark well locations (at x_D = ±1, y_D = 0)
    ax1.plot(1, 0, 'ro', markersize=8, label='Well 1')
    ax1.plot(-1, 0, 'bo', markersize=8, label='Well 2')
    ax1.legend()
    
    # Plot 2: ψ_D as function of x_D, y_D
    ax2 = axes[0, 1]
    contour2 = ax2.contourf(X, Y, Psi, levels=50, cmap='twilight')
    ax2.set_xlabel(r'$x_D$', fontsize=12)
    ax2.set_ylabel(r'$y_D$', fontsize=12)
    ax2.set_title(r'$\psi_D(x_D, y_D)$', fontsize=13)
    ax2.set_aspect('equal')
    fig.colorbar(contour2, ax=ax2, label=r'$\psi_D$ (rad)')
    ax2.plot(1, 0, 'ro', markersize=8)
    ax2.plot(-1, 0, 'bo', markersize=8)
    
    # Plot 3: Streamlines in (x_D, y_D) plane
    ax3 = axes[0, 2]
    # Calculate stream function for dipole (for visualization)
    # ψ_dipole = y/(x^2+y^2) - y/((x-2)^2+y^2)  # Simplified
    ax3.streamplot(X, Y, Phi, Psi, color='gray', linewidth=0.5, density=1.5)
    ax3.set_xlabel(r'$x_D$', fontsize=12)
    ax3.set_ylabel(r'$y_D$', fontsize=12)
    ax3.set_title('Coordinate Transformation Streamlines', fontsize=13)
    ax3.set_aspect('equal')
    ax3.plot(1, 0, 'ro', markersize=8)
    ax3.plot(-1, 0, 'bo', markersize=8)
    
    # Plot 4: φ_D vs ψ_D scatter (color by radius)
    ax4 = axes[1, 0]
    r = np.sqrt(X**2 + Y**2)
    scatter = ax4.scatter(Phi.ravel(), Psi.ravel(), c=r.ravel(), 
                         cmap='viridis', s=1, alpha=0.6)
    ax4.set_xlabel(r'$\phi_D$', fontsize=12)
    ax4.set_ylabel(r'$\psi_D$ (rad)', fontsize=12)
    ax4.set_title(r'$(\phi_D, \psi_D)$ Space', fontsize=13)
    fig.colorbar(scatter, ax=ax4, label='Radius')
    
    # Plot 5: φ_D along x-axis (y_D = 0)
    ax5 = axes[1, 1]
    y_zero = 0
    phi_x_axis = phi_D(x_vals, y_zero)
    ax5.plot(x_vals, phi_x_axis, 'b-', linewidth=2)
    ax5.set_xlabel(r'$x_D$ (y_D = 0)', fontsize=12)
    ax5.set_ylabel(r'$\phi_D$', fontsize=12)
    ax5.set_title(r'$\phi_D$ along x-axis', fontsize=13)
    ax5.grid(True, alpha=0.3)
    ax5.axvline(x=1, color='r', linestyle='--', alpha=0.5)
    ax5.axvline(x=-1, color='b', linestyle='--', alpha=0.5)
    
    # Plot 6: ψ_D along y-axis (x_D = 0)
    ax6 = axes[1, 2]
    x_zero = 0
    psi_y_axis = psi_D(x_zero, y_vals)
    ax6.plot(y_vals, psi_y_axis, 'r-', linewidth=2)
    ax6.set_xlabel(r'$y_D$ (x_D = 0)', fontsize=12)
    ax6.set_ylabel(r'$\psi_D$ (rad)', fontsize=12)
    ax6.set_title(r'$\psi_D$ along y-axis', fontsize=13)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Coordinate Transformation: (x_D, y_D) → (φ_D, ψ_D)', fontsize=15)
    plt.tight_layout()
    plt.show()
    
    return X, Y, Phi, Psi

def analyze_temperature_field(t_D=5.0, a=1.0, b=0.5, alpha=1.0, beta=1.0, 
                             x_range=(-4, 4), y_range=(-3, 3)):
    """
    Analyze the temperature field in the (x_D, y_D) plane at a given time.
    """
    # Create grid
    x_vals = np.linspace(x_range[0], x_range[1], 150)
    y_vals = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Calculate complete solution at each point
    phi_grid, psi_grid, I_D_grid, T_D_grid = complete_solution(
        t_D, X, Y, a, b, alpha, beta
    )
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Temperature field T_D(x_D, y_D)
    ax1 = axes[0, 0]
    contour1 = ax1.contourf(X, Y, T_D_grid, levels=50, cmap='hot_r', vmin=0, vmax=1)
    ax1.set_xlabel(r'$x_D$', fontsize=12)
    ax1.set_ylabel(r'$y_D$', fontsize=12)
    ax1.set_title(f'Temperature Field at t_D = {t_D}', fontsize=13)
    ax1.set_aspect('equal')
    fig.colorbar(contour1, ax=ax1, label=r'$T_D$')
    
    # Mark well locations
    ax1.plot(1, 0, 'wo', markersize=10, markeredgecolor='k')
    ax1.plot(-1, 0, 'wo', markersize=10, markeredgecolor='k')
    ax1.text(1.1, 0.1, 'Production', fontsize=10, color='white')
    ax1.text(-1.3, 0.1, 'Injection', fontsize=10, color='white')
    
    # Plot 2: I_D field
    ax2 = axes[0, 1]
    contour2 = ax2.contourf(X, Y, I_D_grid, levels=50, cmap='viridis')
    ax2.set_xlabel(r'$x_D$', fontsize=12)
    ax2.set_ylabel(r'$y_D$', fontsize=12)
    ax2.set_title(r'$I_D(x_D, y_D)$ Field', fontsize=13)
    ax2.set_aspect('equal')
    fig.colorbar(contour2, ax=ax2, label=r'$I_D$')
    ax2.plot(1, 0, 'wo', markersize=8, markeredgecolor='k')
    ax2.plot(-1, 0, 'wo', markersize=8, markeredgecolor='k')
    
    # Plot 3: Thermal front position (T_D = 0.5 contour)
    ax3 = axes[0, 2]
    # Find contour where T_D = 0.5
    T_contour = ax3.contour(X, Y, T_D_grid, levels=[0.1, 0.3, 0.5, 0.7, 0.9], 
                           colors=['blue', 'green', 'red', 'orange', 'purple'], 
                           linewidths=2)
    ax3.clabel(T_contour, inline=True, fontsize=9)
    ax3.set_xlabel(r'$x_D$', fontsize=12)
    ax3.set_ylabel(r'$y_D$', fontsize=12)
    ax3.set_title(f'Thermal Fronts at t_D = {t_D}', fontsize=13)
    ax3.set_aspect('equal')
    ax3.plot(1, 0, 'ko', markersize=8, label='Production')
    ax3.plot(-1, 0, 'ko', markersize=8, label='Injection')
    ax3.legend()
    
    # Plot 4: Temperature along x-axis (y_D = 0)
    ax4 = axes[1, 0]
    y_zero = 0
    T_x_axis = complete_solution(t_D, x_vals, y_zero, a, b, alpha, beta)[3]
    ax4.plot(x_vals, T_x_axis, 'b-', linewidth=2)
    ax4.set_xlabel(r'$x_D$ (y_D = 0)', fontsize=12)
    ax4.set_ylabel(r'$T_D$', fontsize=12)
    ax4.set_title(f'Temperature along x-axis', fontsize=13)
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=1, color='r', linestyle='--', alpha=0.5, label='Production well')
    ax4.axvline(x=-1, color='b', linestyle='--', alpha=0.5, label='Injection well')
    ax4.legend()
    
    # Plot 5: Temperature along y-axis (x_D = 0)
    ax5 = axes[1, 1]
    x_zero = 0
    T_y_axis = complete_solution(t_D, x_zero, y_vals, a, b, alpha, beta)[3]
    ax5.plot(y_vals, T_y_axis, 'r-', linewidth=2)
    ax5.set_xlabel(r'$y_D$ (x_D = 0)', fontsize=12)
    ax5.set_ylabel(r'$T_D$', fontsize=12)
    ax5.set_title(f'Temperature along y-axis', fontsize=13)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Parameters info
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    param_text = f'Parameters:\n'
    param_text += f't_D = {t_D}\n'
    param_text += f'a = {a}, b = {b}\n'
    param_text += f'α = {alpha}, β = {beta}\n'
    param_text += f'a/b = {a/b:.2f}\n\n'
    
    # Calculate some key metrics
    # Temperature at production well (x_D=1, y_D=0)
    T_prod = complete_solution(t_D, 1, 0, a, b, alpha, beta)[3]
    param_text += f'T at production well:\nT_D = {T_prod:.4f}\n\n'
    
    # Thermal breakthrough time for T_D = 0.5 at production well
    # (simplified calculation)
    I_D_prod = complete_solution(t_D, 1, 0, a, b, alpha, beta)[2]
    param_text += f'I_D at production well:\nI_D = {I_D_prod:.4f}\n\n'
    
    # Calculate approximate breakthrough time for T_D = 0.5
    if T_prod < 0.5:
        param_text += f'Breakthrough (T_D=0.5)\nnot yet reached'
    else:
        # Simple estimate
        t_break_est = I_D_prod + (beta**2 * I_D_prod**2) / (alpha * 1.386**2)
        param_text += f'Est. breakthrough time:\nt_D ≈ {t_break_est:.2f}'
    
    ax6.text(0.5, 0.5, param_text, fontsize=11, ha='center', va='center',
             transform=ax6.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle(f'Geothermal Heat Extraction Analysis (t_D = {t_D})', fontsize=15)
    plt.tight_layout()
    plt.show()
    
    return X, Y, T_D_grid, I_D_grid

def time_evolution_animation(x_D=1.0, y_D=0.0, a=1.0, b=0.5, alpha=1.0, beta=1.0, 
                            t_max=20, num_points=100):
    """
    Plot temperature evolution over time at a specific location.
    """
    # Time array
    t_D_vals = np.linspace(0, t_max, num_points)
    
    # Calculate complete solution over time
    phi_vals, psi_vals, I_D_vals, T_D_vals = complete_solution(
        t_D_vals, x_D, y_D, a, b, alpha, beta
    )
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    
    # Plot 1: Temperature vs time
    ax1 = axes[0, 0]
    ax1.plot(t_D_vals, T_D_vals, 'b-', linewidth=2)
    ax1.set_xlabel('Dimensionless Time (t_D)', fontsize=12)
    ax1.set_ylabel('Dimensionless Temperature (T_D)', fontsize=12)
    ax1.set_title(f'Temperature Evolution at (x_D={x_D}, y_D={y_D})', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Mark I_D value
    I_D_constant = I_D_vals[0] if np.isscalar(I_D_vals) else I_D_vals
    ax1.axvline(x=I_D_constant, color='r', linestyle='--', alpha=0.7, 
                label=f'I_D = {I_D_constant:.3f}')
    ax1.legend()
    
    # Plot 2: Temperature derivative (heating/cooling rate)
    ax2 = axes[0, 1]
    dT_dt = np.gradient(T_D_vals, t_D_vals[1] - t_D_vals[0])
    ax2.plot(t_D_vals, dT_dt, 'g-', linewidth=2)
    ax2.set_xlabel('Dimensionless Time (t_D)', fontsize=12)
    ax2.set_ylabel('dT_D/dt_D', fontsize=12)
    ax2.set_title('Temperature Change Rate', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=I_D_constant, color='r', linestyle='--', alpha=0.7)
    
    # Plot 3: φ_D and ψ_D values
    ax3 = axes[1, 0]
    ax3.plot(t_D_vals, phi_vals, 'b-', linewidth=2, label=r'$\phi_D$')
    ax3.plot(t_D_vals, psi_vals, 'r-', linewidth=2, label=r'$\psi_D$')
    ax3.set_xlabel('Dimensionless Time (t_D)', fontsize=12)
    ax3.set_ylabel('Coordinate Values', fontsize=12)
    ax3.set_title('Dimensionless Coordinates', fontsize=13)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Plot 4: Parameter relationships
    ax4 = axes[1, 1]
    
    # Calculate the argument of erfc function
    mask = t_D_vals > I_D_constant
    argument = np.zeros_like(t_D_vals)
    argument[mask] = beta * I_D_constant / np.sqrt(alpha * (t_D_vals[mask] - I_D_constant))
    
    ax4.plot(t_D_vals, argument, 'm-', linewidth=2)
    ax4.set_xlabel('Dimensionless Time (t_D)', fontsize=12)
    ax4.set_ylabel('erfc Argument', fontsize=12)
    ax4.set_title('Argument of erfc Function', fontsize=13)
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=I_D_constant, color='r', linestyle='--', alpha=0.7)
    
    # Add parameter info
    param_text = f'Parameters:\n'
    param_text += f'a={a}, b={b}, a/b={a/b:.2f}\n'
    param_text += f'α={alpha}, β={beta}\n'
    param_text += f'Position: x_D={x_D}, y_D={y_D}\n'
    param_text += f'φ_D={phi_vals[0]:.3f}, ψ_D={psi_vals[0]:.3f} rad\n'
    param_text += f'I_D={I_D_constant:.3f}'
    
    plt.figtext(0.02, 0.02, param_text, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('Time Evolution Analysis', fontsize=15)
    plt.tight_layout()
    plt.show()
    
    return t_D_vals, T_D_vals

# Main execution
if __name__ == "__main__":
    print("=" * 80)
    print("COMPLETE GEOTHERMAL HEAT EXTRACTION SOLUTION")
    print("=" * 80)
    
    # Display all formulas
    print("\nMathematical Model:")
    print("1. Coordinate Transformation:")
    print("   φ_D = 0.5 * ln[((x_D+1)² + y_D²) / ((x_D-1)² + y_D²)]")
    print("   ψ_D = arctan[2*y_D / (1 - x_D² - y_D²)]")
    print("\n2. Dimensionless Parameter:")
    print("   I_D = (2π/3)*(a²/b²)*[1 - A * (1 + B)]")
    print("   where A = sinh(φ_D)/(cosh(φ_D)+cosh(ψ_D))")
    print("         B = cos(ψ_D)/(cos(φ_D)+cos(ψ_D))")
    print("\n3. Temperature Solution:")
    print("   T_D = u(t_D - I_D) * erfc(β*I_D / √[α*(t_D - I_D)])")
    print()
    
    # Example 1: Single point calculation
    print("=" * 80)
    print("EXAMPLE 1: SINGLE POINT CALCULATION")
    print("=" * 80)
    
#    t_D_example = 5.0
#    x_D_example, y_D_example = 1.0, 0.5
#    a_example, b_example = 1.0, 0.5
#    alpha_example, beta_example = 0.8, 1.2
    t_D_example = 3.1526e+12
    x_D_example, y_D_example = 1.0, 0.0
    a_example, b_example = 100., 0.01
    alpha_example, beta_example = 8e-6, 4e-6
    
    phi, psi, I_D, T_D = complete_solution(
        t_D_example, x_D_example, y_D_example,
        a_example, b_example, alpha_example, beta_example
    )
    
    print(f"\nInput Parameters:")
    print(f"  Position: x_D = {x_D_example}, y_D = {y_D_example}")
    print(f"  Time: t_D = {t_D_example}")
    print(f"  Geometry: a = {a_example}, b = {b_example}")
    print(f"  Thermal: α = {alpha_example}, β = {beta_example}")
    
    print(f"\nIntermediate Results:")
    print(f"  φ_D = {phi:.6f}")
    print(f"  ψ_D = {psi:.6f} rad ({np.degrees(psi):.2f}°)")
    print(f"  I_D = {I_D:.6f}")
    
    print(f"\nFinal Result:")
    print(f"  T_D = {T_D:.6f}")
    
    # Check if thermal breakthrough has occurred
    if t_D_example > I_D:
        print(f"  Status: Thermal breakthrough has occurred (t_D > I_D)")
        print(f"  Time since breakthrough: Δt = {t_D_example - I_D:.3f}")
    else:
        print(f"  Status: No breakthrough yet (t_D ≤ I_D)")
        print(f"  Time until breakthrough: Δt = {I_D - t_D_example:.3f}")
    
    # Example 2: Multiple points along a line
    print("\n" + "=" * 80)
    print("EXAMPLE 2: TEMPERATURE PROFILE ALONG X-AXIS")
    print("=" * 80)
    
    t_D_profile = 3.0
    x_D_profile = np.linspace(-2, 2, 21)
    y_D_profile = 0.0  # Along x-axis
    
#    phi_profile, psi_profile, I_D_profile, T_D_profile = complete_solution(
#        t_D_profile, x_D_profile, y_D_profile,
#        a_example, b_example, alpha_example, beta_example
#    )
    
    print(f"\nTemperature along x-axis at t_D = {t_D_profile}:")
    print("-" * 60)
    print(f"{'x_D':>8} {'φ_D':>12} {'ψ_D (rad)':>12} {'I_D':>12} {'T_D':>12}")
    print("-" * 60)
    
#    for i in range(len(x_D_profile)):
#        print(f"{x_D_profile[i]:8.2f} {phi_profile[i]:12.6f} {psi_profile[i]:12.6f} "
#              f"{I_D_profile[i]:12.6f} {T_D_profile[i]:12.6f}")
    
    print("-" * 60)
    
    # Example 3: Visualization
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS...")
    print("=" * 80)
    
    # Visualization 1: Coordinate transformation
    print("\n1. Visualizing coordinate transformation...")
    X_grid, Y_grid, Phi_grid, Psi_grid = plot_coordinate_transformation(
        x_range=(-3, 3), y_range=(-3, 3)
    )
    
    # Visualization 2: Temperature field at specific time
    print("\n2. Analyzing temperature field...")
    T_D_time = 4.0
    X_temp, Y_temp, T_D_temp, I_D_temp = analyze_temperature_field(
        t_D=T_D_time,
        a=1.0,
        b=0.5,
        alpha=1.0,
        beta=1.0,
        x_range=(-4, 4),
        y_range=(-3, 3)
    )
    
    # Visualization 3: Time evolution at a point
    print("\n3. Plotting time evolution...")
    t_vals, T_vals = time_evolution_animation(
        x_D=1.0,  # Production well location
        y_D=0.0,
        a=1.0,
        b=0.5,
        alpha=1.0,
        beta=1.0,
        t_max=15,
        num_points=200
    )
    
    # Example 4: Sensitivity analysis
    print("\n" + "=" * 80)
    print("SENSITIVITY ANALYSIS")
    print("=" * 80)
    
    # Analyze sensitivity to position
    positions = [
        (1.0, 0.0),   # At production well
        (0.0, 0.0),   # Midpoint between wells
        (0.0, 1.0),   # Above midpoint
        (2.0, 0.0),   # Beyond production well
    ]
    
    t_D_sens = 5.0
    
    print(f"\nTemperature at different positions (t_D = {t_D_sens}):")
    print("-" * 80)
    print(f"{'Position (x_D,y_D)':>20} {'φ_D':>12} {'ψ_D (deg)':>12} {'I_D':>12} {'T_D':>12}")
    print("-" * 80)
    
    for x_pos, y_pos in positions:
        phi_sens, psi_sens, I_D_sens, T_D_sens = complete_solution(
            t_D_sens, x_pos, y_pos,
            a_example, b_example, alpha_example, beta_example
        )
        print(f"({x_pos:4.1f}, {y_pos:4.1f}){'':>10}{phi_sens:12.6f} "
              f"{np.degrees(psi_sens):12.2f} {I_D_sens:12.6f} {T_D_sens:12.6f}")
    
    print("-" * 80)