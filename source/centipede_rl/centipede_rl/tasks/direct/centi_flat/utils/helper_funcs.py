import math
import numpy as np
def wrap_to_2pi(angle):
    return angle % (2 * math.pi)

def wrap_to_pi(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def nearest_equivalent_angle(current, target_0to2pi):
    """Map target in [0, 2π) to the closest equivalent angle to 'current'."""
    return current + wrap_to_pi(target_0to2pi - current)

def get_F_tw1(t,body_amp):
    return body_amp * np.cos(t)

def get_F_tw2(t,body_amp):
    return body_amp * np.sin(t)

def alpha(F_tw1,shape_basis1,F_tw2,shape_basis2):
  # compute body undulation angle alpha
    return (F_tw1*shape_basis1+F_tw2*shape_basis2)

def alpha_v(t, v_amp,xis, N, shape_tile):

    x_values = np.arange(0, xis * (N-1), xis) * 2 * np.pi + shape_tile + t
    result = v_amp * np.cos(2 * x_values + np.pi / 2) 
    return result



def F_Leg(Aleg, time, dutyf):
    time = np.mod(time, 2*np.pi)




    if time < 2*np.pi*dutyf:
        leg_act_col = Aleg * np.cos(time/(2*dutyf))
    else:
        leg_act_col = Aleg * np.cos((time-2*np.pi)/(2*(1-dutyf)))
    return leg_act_col

def F_leg_act(time, dutyf):
    leg_act_col = np.zeros_like(time)
    time = np.mod(time, 2*np.pi)
    if time < 2*np.pi*dutyf:
        leg_act_col = 2
    else:
        leg_act_col = 0
    return leg_act_col



def get_beta(c, num_leg, Aleg, dutyf, lfs, symmode):
    beta = np.zeros((1, num_leg))

    for ind in range(1, num_leg + 1):
        beta[:, ind - 1] = F_Leg(Aleg, c + (ind - 1) * lfs * 2 * np.pi, dutyf)

    return beta


def get_act(c,lfs,num_leg,dutyf):
  act=np.zeros((1,num_leg))
  for ind in range(0,num_leg):
      act[:,ind]=F_leg_act(c+(ind-1)*lfs*2*np.pi,dutyf)
  return act



def biased_angle_from_time(t, T=2.0, frac=0.8):
    """
    Map time t -> angle theta in [-pi, pi], with `frac` of the cycle
    spent in the interval [-pi/6, 0] and the rest distributed uniformly
    across [-pi, -pi/6] and [0, pi].

    Parameters
    ----------
    t : float
        Time (seconds).
    T : float
        Period (seconds). Default = 2.0.
    frac : float
        Fraction of time spent in [-pi/6, 0] (0 < frac < 1).
    """
    pi = math.pi
    u = (t % T) / T   # normalized time in [0,1)

    # Interval lengths
    L1 = 5*pi/6      # [-pi, -pi/6]
    L2 = pi/6        # [-pi/6, 0]
    L3 = pi          # [0, pi]

    # Assign time mass
    m2 = frac
    m_rest = 1 - frac
    # split remaining proportionally by interval lengths
    total_rest_len = L1 + L3
    m1 = m_rest * (L1 / total_rest_len)
    m3 = m_rest * (L3 / total_rest_len)

    # PDF values (constant on each segment)
    a = m2 / L2
    b1 = m1 / L1
    b3 = m3 / L3

    # CDF cutoffs
    u1 = m1
    u2 = m1 + m2

    # Inverse CDF
    if u < u1:
        theta = -pi + u / b1
    elif u < u2:
        theta = -pi/6 + (u - m1) / a
    else:
        theta = 0 + (u - (m1 + m2)) / b3

    # Wrap to [-pi, pi] for safety
    return (theta + pi) % (2*pi) - pi

def get_wheel_angle(c, lfs, num_leg):
    wheel_angle = np.zeros((1, num_leg))
    for ind in range(0, num_leg):
        wheel_angle[:, ind] = c + (ind - 1) * lfs * 2 * np.pi 
    return wheel_angle