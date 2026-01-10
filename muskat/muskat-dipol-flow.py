import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def velocity_potential(x, y, U, Q, h, a):
    """
    Calculate velocity potential for wellbore doublet.
    
    Parameters:
    x, y : array_like - Grid coordinates
    U : float - Uniform flow velocity
    Q : float - Flow rate (positive for injection at -a, extraction at +a)
    h : float - Formation thickness
    a : float - Half distance between wells
    
    Returns:
    phi : array_like - Velocity potential
    """
    r1 = np.sqrt((x + a)**2 + y**2)
    r2 = np.sqrt((x - a)**2 + y**2)
    phi = -U * x + (Q / (2 * np.pi * h)) * np.log(r1 / r2)
    return phi

def stream_function(x, y, U, Q, h, a):
    """
    Calculate stream function for wellbore doublet.
    
    Parameters:
    x, y : array_like - Grid coordinates
    U : float - Uniform flow velocity
    Q : float - Flow rate
    h : float - Formation thickness
    a : float - Half distance between wells
    
    Returns:
    psi : array_like - Stream function
    """
    # Using arctan2 to handle quadrant correctly
    theta1 = np.arctan2(y, x + a)
    theta2 = np.arctan2(y, x - a)
    psi = -U * y + (Q / (2 * np.pi * h)) * (theta1 - theta2)
    return psi

def velocity_components(x, y, U, Q, h, a):
    """
    Calculate velocity components.
    
    Returns:
    vx, vy : arrays - Velocity components in x and y directions
    """
    r1_sq = (x + a)**2 + y**2
    r2_sq = (x - a)**2 + y**2
    
    vx = U - (Q / (2 * np.pi * h)) * ((x + a) / r1_sq - (x - a) / r2_sq)
    vy = - (Q / (2 * np.pi * h)) * (y / r1_sq - y / r2_sq)
    
    return vx, vy

def plot_potential_and_streamlines(U=1.0, Q=10.0, h=5.0, a=10.0, 
                                   xlim=(-30, 30), ylim=(-30, 30), 
                                   n_points=200):
    """
    Create comprehensive plot of potential and streamlines.
    """
    # Create grid
    x = np.linspace(xlim[0], xlim[1], n_points)
    y = np.linspace(ylim[0], ylim[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    # Avoid division by zero at well locations
    mask1 = ((X + a)**2 + Y**2) < 0.1
    mask2 = ((X - a)**2 + Y**2) < 0.1
    mask = mask1 | mask2
    
    # Calculate potential and stream function
    phi = velocity_potential(X, Y, U, Q, h, a)
    psi = stream_function(X, Y, U, Q, h, a)
    
    # Mask out near wells for cleaner plots
    phi_masked = np.ma.array(phi, mask=mask)
    psi_masked = np.ma.array(psi, mask=mask)
    
    # Calculate stagnation points
    lambda_param = Q / (np.pi * h * a * U)
    if lambda_param > -1:
        x_stag = -a * np.sqrt(1 + lambda_param)
        y_stag = 0
        stagnation_points = [(x_stag, y_stag)]
        if lambda_param >= 0:
            x_stag2 = a * np.sqrt(1 + lambda_param)
            stagnation_points.append((x_stag2, 0))
    else:
        stagnation_points = []
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Velocity Potential Contours
    ax1 = axes[0, 0]
    contour1 = ax1.contour(X, Y, phi_masked, levels=30, colors='blue', linewidths=0.5)
    ax1.contourf(X, Y, phi_masked, levels=30, cmap=cm.Blues, alpha=0.7)
    ax1.set_title('Velocity Potential ($\Phi$) Contours')
    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Stream Function Contours
    ax2 = axes[0, 1]
    contour2 = ax2.contour(X, Y, psi_masked, levels=30, colors='red', linewidths=0.5)
    ax2.contourf(X, Y, psi_masked, levels=30, cmap=cm.Reds_r, alpha=0.7)
    ax2.set_title('Stream Function ($\Psi$) Contours')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Flow Net (Potential and Streamlines)
    ax3 = axes[0, 2]
    # Streamlines (in black)
    ax3.streamplot(X, Y, *velocity_components(X, Y, U, Q, h, a), 
                   color='black', linewidth=0.5, density=2, arrowsize=0.5)
    # Equipotential lines (in blue)
    ax3.contour(X, Y, phi_masked, levels=20, colors='blue', linewidths=0.5, alpha=0.7)
    ax3.set_title('Flow Net (Streamlines & Equipotentials)')
    ax3.set_xlabel('x [m]')
    ax3.set_ylabel('y [m]')
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: 3D Surface of Velocity Potential
    ax4 = axes[1, 0]
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    surf = ax4.plot_surface(X, Y, phi, cmap=cm.viridis, 
                           alpha=0.8, linewidth=0, antialiased=True)
    ax4.set_title('3D Surface: Velocity Potential')
    ax4.set_xlabel('x [m]')
    ax4.set_ylabel('y [m]')
    ax4.set_zlabel('$\Phi$ [m²/s]')
    
    # Plot 5: Velocity Magnitude
    ax5 = axes[1, 1]
    vx, vy = velocity_components(X, Y, U, Q, h, a)
    velocity_magnitude = np.sqrt(vx**2 + vy**2)
    velocity_magnitude = np.ma.array(velocity_magnitude, mask=mask)
    
    im = ax5.imshow(velocity_magnitude, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], 
                    origin='lower', cmap=cm.hot, alpha=0.8)
    ax5.streamplot(X, Y, vx, vy, color='white', linewidth=0.5, density=1.5, arrowsize=0.5)
    ax5.set_title('Velocity Magnitude & Streamlines')
    ax5.set_xlabel('x [m]')
    ax5.set_ylabel('y [m]')
    ax5.set_aspect('equal')
    plt.colorbar(im, ax=ax5, label='Velocity Magnitude [m/s]')
    
    # Plot 6: Capture Zone and Well Locations
    ax6 = axes[1, 2]
    # Plot streamlines
    ax6.streamplot(X, Y, vx, vy, color='gray', linewidth=0.5, density=1.5)
    
    # Highlight dividing streamline (capture zone boundary)
    # The dividing streamline has value Q/(2h)
    dividing_value = Q / (2 * h)
    contour6 = ax6.contour(X, Y, psi, levels=[dividing_value], 
                          colors='red', linewidths=2, linestyles='--')
    
    # Mark wells
    ax6.plot(-a, 0, 'bo', markersize=10, label='Injection Well (Source)')
    ax6.plot(a, 0, 'ro', markersize=10, label='Production Well (Sink)')
    
    # Mark stagnation points
    for stag_x, stag_y in stagnation_points:
        ax6.plot(stag_x, stag_y, 'g*', markersize=15, label='Stagnation Point')
    
    # Add uniform flow arrow
    ax6.arrow(xlim[0] + 5, 0, 10, 0, head_width=2, head_length=2, 
              fc='green', ec='green', label='Uniform Flow')
    
    ax6.set_title('Capture Zone & Well Configuration')
    ax6.set_xlabel('x [m]')
    ax6.set_ylabel('y [m]')
    ax6.set_aspect('equal')
    ax6.grid(True, alpha=0.3)
    ax6.legend(loc='upper right', fontsize=8)
    
    # Add text box with parameters
    param_text = (f'Parameters:\n'
                  f'U = {U:.1f} m/s\n'
                  f'Q = {Q:.1f} m³/s\n'
                  f'h = {h:.1f} m\n'
                  f'2a = {2*a:.1f} m\n'
                  f'λ = Q/(πh a U) = {lambda_param:.3f}')
    
    fig.text(0.02, 0.02, param_text, fontsize=10, 
             verticalalignment='bottom', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig, (phi, psi, vx, vy)

def plot_interactive_comparison():
    """
    Create interactive comparison of different parameter sets.
    """
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button
    
    # Default parameters
    U_default = 1.0
    Q_default = 10.0
    h_default = 5.0
    a_default = 10.0
    
    # Create figure and axes
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(left=0.1, bottom=0.3, right=0.9, top=0.95)
    
    # Create mesh
    xlim = (-30, 30)
    ylim = (-30, 30)
    n_points = 100
    x = np.linspace(xlim[0], xlim[1], n_points)
    y = np.linspace(ylim[0], ylim[1], n_points)
    X, Y = np.meshgrid(x, y)
    
    # Initial plot
    def update_plots(U=U_default, Q=Q_default, h=h_default, a=a_default):
        for ax in axes.flat:
            ax.clear()
        
        # Calculate values
        phi = velocity_potential(X, Y, U, Q, h, a)
        psi = stream_function(X, Y, U, Q, h, a)
        vx, vy = velocity_components(X, Y, U, Q, h, a)
        
        # Plot 1: Streamlines
        axes[0, 0].streamplot(X, Y, vx, vy, color='blue', linewidth=0.5, density=2)
        axes[0, 0].set_title('Streamlines')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        axes[0, 0].set_aspect('equal')
        axes[0, 0].plot(-a, 0, 'ro', markersize=8)
        axes[0, 0].plot(a, 0, 'bo', markersize=8)
        
        # Plot 2: Velocity Potential
        contour = axes[0, 1].contourf(X, Y, phi, levels=20, cmap=cm.RdYlBu_r)
        axes[0, 1].set_title('Velocity Potential')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        axes[0, 1].set_aspect('equal')
        plt.colorbar(contour, ax=axes[0, 1])
        
        # Plot 3: Stream Function
        contour = axes[1, 0].contourf(X, Y, psi, levels=20, cmap=cm.PiYG)
        axes[1, 0].set_title('Stream Function')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')
        axes[1, 0].set_aspect('equal')
        plt.colorbar(contour, ax=axes[1, 0])
        
        # Plot 4: Velocity Magnitude
        vel_mag = np.sqrt(vx**2 + vy**2)
        contour = axes[1, 1].contourf(X, Y, vel_mag, levels=20, cmap=cm.hot)
        axes[1, 1].set_title('Velocity Magnitude')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')
        axes[1, 1].set_aspect('equal')
        plt.colorbar(contour, ax=axes[1, 1])
        
        fig.suptitle(f'U={U:.1f}, Q={Q:.1f}, h={h:.1f}, a={a:.1f}', fontsize=12)
        
    update_plots()
    
    # Create sliders
    axcolor = 'lightgoldenrodyellow'
    ax_U = plt.axes([0.15, 0.2, 0.65, 0.03], facecolor=axcolor)
    ax_Q = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor=axcolor)
    ax_h = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor=axcolor)
    ax_a = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor=axcolor)
    
    s_U = Slider(ax_U, 'U', 0.1, 5.0, valinit=U_default)
    s_Q = Slider(ax_Q, 'Q', 1.0, 50.0, valinit=Q_default)
    s_h = Slider(ax_h, 'h', 1.0, 20.0, valinit=h_default)
    s_a = Slider(ax_a, 'a', 5.0, 20.0, valinit=a_default)
    
    def update(val):
        update_plots(s_U.val, s_Q.val, s_h.val, s_a.val)
        fig.canvas.draw_idle()
    
    s_U.on_changed(update)
    s_Q.on_changed(update)
    s_h.on_changed(update)
    s_a.on_changed(update)
    
    # Add reset button
    resetax = plt.axes([0.85, 0.025, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
    
    def reset(event):
        s_U.reset()
        s_Q.reset()
        s_h.reset()
        s_a.reset()
    
    button.on_clicked(reset)
    
    return fig

# Example usage:
if __name__ == "__main__":
    # Set parameters
    U = 0.01    # Uniform flow velocity [m/s]
    Q = 10.0   # Flow rate [m³/s] (positive: injection at -a, extraction at +a)
    h = 5.0    # Formation thickness [m]
    a = 10.0   # Half distance between wells [m]
    
    # Generate comprehensive plot
    fig, results = plot_potential_and_streamlines(U, Q, h, a)
    plt.show()
    
    # For interactive plot (uncomment to use):
    # fig_interactive = plot_interactive_comparison()
    # plt.show()
    
    # Extract results for further analysis
    phi, psi, vx, vy = results
    
    # Calculate and print stagnation points
    lambda_param = Q / (np.pi * h * a * U)
    if lambda_param > -1:
        x_stag = -a * np.sqrt(1 + lambda_param)
        print(f"Stagnation point between wells: ({x_stag:.2f}, 0.00)")
        if lambda_param >= 0:
            x_stag2 = a * np.sqrt(1 + lambda_param)
            print(f"Stagnation point upstream: ({x_stag2:.2f}, 0.00)")
    else:
        print("No stagnation points between wells (λ <= -1)")