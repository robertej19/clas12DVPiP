

for x in range(0, len(x_bins) - (x_width-1), x_stride):
    for q in range(0, len(q_bins) - (q_width-1), q_stride):
        for t in range(0, len(t_bins) - (t_width-1), t_stride):
            for phi in range(0, len(phi_bins), phi_stride):
                x_range = x_bins[x:x+x_width]
                q_range = q_bins[q:q+q_width]
                t_range = t_bins[t:t+t_width]
                # For phi, use modulo operation to wrap indices around
                phi_indices = [(phi + i) % len(phi_bins) for i in range(phi_width)]
                phi_range = phi_bins[phi_indices]