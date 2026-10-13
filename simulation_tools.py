import numpy as np
from scipy.integrate import solve_ivp


###########
# HELPERS #
###########

# Numbers
def is_num(var, name):
    if not(isinstance(var, int) or isinstance(var, float) or isinstance(var, np.number)):
        raise TypeError(f"'{name}' should be of an integer type, but is of type {type(var)} instead.")
    return

def is_pos_num(var, name):
    is_num(var, name)
    if var < 0:
        raise ValueError(f"'{name}' should be positive.")
    return

def is_strict_pos_num(var, name):
    is_num(var, name)
    if var <= 0:
        raise ValueError(f"'{name}' should be strictly positive.")
    return

def is_int(var, name):
    if not(isinstance(var, int) or isinstance(var, np.integer)):
        raise TypeError(f"'{name}' should be of an integer type, but is of type {type(var)} instead.")
    return

def is_pos_int(var, name):
    is_int(var, name)
    if var < 0:
        raise ValueError(f"'{name}' should be positive.")
    return

def is_strict_pos_int(var, name):
    is_int(var, name)
    if var <= 0:
        raise ValueError(f"'{name}' should be strictly positive.")
    return

def is_bool(var, name):
    if not(isinstance(var, bool) or isinstance(var, np.bool)):
        raise TypeError(f"'{name}' should be of a boolean type, but is of type {type(var)} instead.")
    return

# Matrices and vectors
def is_mat(mat, name):
    if not isinstance(mat, np.ndarray):
        raise TypeError(f"'{name}' should be of type numpy.ndarray, but is of type {type(mat)} instead.")
    if len(mat.shape) != 2:
        raise ValueError(f"'{name}' should be 2-dimensional, but is of shape {mat.shape} instead.")
    if not isinstance(mat[0, 0], np.number):
        raise TypeError(f"Elements of '{name}' should be numerical numpy scalars, but are of type {type(mat[0, 0])} instead.")
    return

def is_square_mat(mat, name):
    is_mat(mat, name)
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"'{name}' should be square, but is of shape {mat.shape} instead.")
    return

def is_vec(vec, name):
    if not isinstance(vec, np.ndarray):
        raise TypeError(f"'{name}' should be of type numpy.ndarray, but is of type {type(vec)} instead.")
    if len(vec.shape) != 1:
        raise ValueError(f"'{name}' should be 1-dimensional, but is of shape {vec.shape} instead.")
    if not isinstance(vec[0], np.number):
        raise TypeError(f"Elements of '{name}' should be numerical numpy scalars, but are of type {type(vec[0])} instead.")
    return

# Arrays of matrices and vectors
def is_mat_arr(arr, name):
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"'{name}' should be of type numpy.ndarray, but is of type {type(arr)} instead.")
    if len(arr.shape) != 3:
        raise ValueError(f"'{name}' should be 3-dimensional, but is of shape {arr.shape} instead.")
    if not isinstance(arr[0, 0, 0], np.number):
        raise TypeError(f"Elements of '{name}' should be numerical numpy scalars, but are of type {type(arr[0, 0, 0])} instead.")
    return

def is_square_mat_arr(arr, name):
    is_mat_arr(arr, name)
    if arr.shape[1] != arr.shape[2]:
        raise ValueError(f"Elements along axis 0 of '{name}' should be square matrices, but are of shape {arr[0].shape} instead.")
    return

# Array-likes of specified size
def is_arr_like_size(arr, name, size):
    if not(isinstance(arr, np.ndarray) or isinstance(arr, list) or isinstance(arr, tuple)):
        raise TypeError(f"'{name}' should be of type numpy.ndarray, list or tuple but is of type {type(arr)} instead.")
    if len(arr) != size:
        raise ValueError(f"'{name}' should have length {size} but is of length {len(arr)} instead.")
    return


###########
# SYSTEMS #
###########

class _NonlinearTimeInvariant():
    def __init__(self, f=None, n=None, m=None, v=None, g=None, p=None, w=None, x0=None):
        # Handle f
        if f is None:
            self.f = None
            self.n = None
            self.m = None
            self.v = None
            self.g = None
            self.p = None
            self.w = None
            self.x = None
            self.system_defined = False
            self.state_defined = False
            self.input = False
            self.output = False
            self.g_input = False
            self.process_noise = False
            self.measurement_noise = False
        else:
            if not callable(f):
                raise TypeError("'f' should be either None or a callable function.")
            f_argcount = f.__code__.co_argcount
            if f_argcount == 1:
                self.input = False
            elif f_argcount == 2:
                self.input = True
            else:
                raise ValueError(f"'f' should take 1 or 2 arguments (not counting *args and **kwargs) but it takes {f_argcount} instead.")
            self.f = f
            self.system_defined = True

            # Handle n & m
            if n is None:
                raise TypeError("'n' is None but should be an integer if 'f' is not None.")
            elif (m is None) and self.input:
                raise TypeError("'m' can't be None if 'f' takes 2 arguments.")
            elif (m is None):
                is_int(n, "n")
                self.n = n
                self.m = None
            else:
                is_int(n, "n")
                is_int(m, "m")
                self.n = n
                self.m = m
            
            # Handle v
            if v is None:
                self.v = None
                self.process_noise = False
            else:
                if not isinstance(v, _BaseNoise):
                    raise TypeError(f"'v' should be a noise process from statespace.noise.processes, but is of type {type(v)}.")
                if v.n != self.n:
                    raise ValueError(f"'v' produces {v.n}-dimesional noise while the systems has a {self.n}-dimesional state.")
                self.v = v
                self.process_noise = True

            # Handle g
            if g is None:
                self.g = None
                self.p = None
                self.w = None
                self.output = False
                self.g_input = False
                self.measurement_noise = False

            else:
                if not callable(g):
                    raise TypeError("'g' should be either None or a callable function.")
                g_argcount = g.__code__.co_argcount
                if self.input:
                    if g_argcount == 1:
                        self.g_input = False
                    elif g_argcount == 2:
                        self.g_input = True
                    else:
                        raise ValueError(f"'g' should take 1 or 2 arguments (not counting *args and **kwargs) but it takes {g_argcount} instead.")
                else:
                    self.g_input = False
                    if g_argcount != 1:
                        raise ValueError(f"'g' should take 1 argument (not counting *args and **kwargs) but it takes {g_argcount} instead.")
                self.g = g
                self.output = True
            
                # Handle p
                if p is None:
                    raise TypeError("'p' is None but should be an integer if g is not None.")
                else:
                    is_int(p, "p")
                    self.p = p
                
                # Handle w
                if w is None:
                    self.w = None
                    self.measurement_noise = False
                else:
                    if not isinstance(v, _BaseNoise):
                        raise TypeError(f"'v' should be a noise process from statespace.noise.processes, but is of type {type(v)}.")
                    if w.n != self.p:
                        raise ValueError(f"'w' produces {w.n}-dimesional noise while the systems has a {self.p}-dimesional output.")
                    self.w = w
                    self.measurement_noise = True

            # Handle x0
            if x0 is None:
                self.x = None
                self.state_defined = False
            else:
                is_vec(x0, 'x0')
                if x0.size != self.n:
                    raise ValueError(f"'x0' should have size {self.n} but has size {x0.size} instead.")
                self.x = x0
                self.state_defined = True
            
            # Check that f and g accept and return arrays of the right size
            if self.system_defined:
                x_dummy = np.zeros(self.n)
                if self.input:
                    u_dummy = np.zeros(self.m)
                    try:
                        x1_dummy = self.f(x_dummy, u_dummy)
                    except:
                        raise ValueError(f"'f' should take 1D numpy arrays of size {self.n} and {self.m} as first and second argument.")
                else:
                    try:
                        x1_dummy = self.f(x_dummy)
                    except:
                        raise ValueError(f"'f' should take a 1D numpy array of size {self.n} as its argument.")
                is_vec(x1_dummy, "return value of f")
                if x1_dummy.size != self.n:
                    raise ValueError(f"'f' should return a 1D numpy of size {self.n} but returns array of size{x1_dummy.size} instead.")
                
                if self.output:
                    if self.g_input:
                        try:
                            y0_dummy = self.g(x_dummy, u_dummy)
                        except:
                            raise ValueError(f"'g' should take 1D numpy arrays of size {self.n} and {self.m} as first and second argument.")
                    else:
                        try:
                            y0_dummy = self.g(x_dummy)
                        except:
                            raise ValueError(f"'g' should take a 1D numpy array of size {self.n} as its argument.")
                    is_vec(y0_dummy, "return value of g")
                    if y0_dummy.size != self.p:
                        raise ValueError(f"'g' should return a 1D numpy of size {self.p} but returns array of size{x1_dummy.size} instead.")            

        return
    
    def get_state(self):
        return self.x
    
    def set_state(self, x):
        # Check if system is defined
        if not self.system_defined:
            raise RuntimeError("Tried to assign state to nonlinear system but function f is not defined.")
        
        # Check and set state
        if x is None:
            self.x = None
            self.state_defined = False
        else:
            is_vec(x, 'x')
            self.x = x
            self.state_defined = True
        return

class ContinuousNonlinearTimeInvariant(_NonlinearTimeInvariant):
    def __str__(self):
        res = "ContinuousNonlinearTimeInvariant:\n"
        if self.system_defined:
            res = res + "x'(t) = f(x(t)"
            if self.input:
                res = res + ", u(t)"
            res = res + ")"

            if self.output:
                res = res + "\ny(t) = g(x(t)"
                if self.g_input:
                    res = res + ", u(t)"
                res = res + ")"
            
            if self.state_defined:
                res = res + "\nx = " + str(self.x)

        else:
            res = res + "    Not defined"

        return res

    def run(self, T, u=None, update_state=True, num=100, endpoint=False):
        # Check if system and state are defined
        if not self.system_defined:
            raise RuntimeError("Tried to run nonlinear system but function f is not defined.")
        if not self.state_defined:
            raise RuntimeError("Tried to run nonlinear system but state x is not defined. The state can be defined using set_state() or during initialization.")
        
        # Check input arguments
        is_bool(update_state, "update_state")
        is_bool(endpoint, "endpoint")
        is_num(T, "T")
        if T <= 0:
            raise ValueError("'T' should be a positive number.")
        is_int(num, "num")
        if num <= 0:
            raise ValueError("'num' should be a positive integer.")

        if u is None:
            run_with_input = False
            if self.input:
                raise RuntimeError("Tried to run nonlinear system without input 'u' but system is defined as x_k+1 = f(x_k, u_k).")
        else:
            run_with_input = True
            if not self.input:
                raise RuntimeError("Tried to run nonlinear system with input 'u' but system is defined as x_k+1 = f(x_k).")
            
            if not callable(u):
                raise TypeError("'u' should be either None or a callable function.")
            u_argcount = u.__code__.co_argcount
            if u_argcount != 1:
                raise ValueError(f"'u' should be a function of one variable, like u(t), but takes {u_argcount} arguments.")
            
            try:
                ut = u(0.0)
                is_vec(ut, "output of u")
            except:
                raise ValueError("'u' should accept scalar floats as input.")
            if ut.size != self.p:
                raise ValueError(f"'u' should give result of length {self.p} but result has length {ut.size} instead.")
        
        # Define initial value problem
        t_eval = np.linspace(0, T, num=num, endpoint=endpoint)

        if run_with_input:
            if self.process_noise:
                v_samples = self.v.sample(k=num) # can't generate noise in f_ivp cause than solver might not converge
                def f_ivp(t, x):
                    ut = u(t)
                    vt = v_samples[:, np.argmax(t_eval > t)]
                    return self.f(x, ut) + vt

            else:
                def f_ivp(t, x):
                    ut = u(t)
                    return self.f(x, ut)
            
        else:
            if self.process_noise:
                v_samples = self.v.sample(k=num)
                def f_ivp(t, x):
                    vt = v_samples[:, np.argmax(t_eval > t)]
                    return self.f(x) + vt
            else:
                def f_ivp(t, x):
                    return self.f(x)
        
        # Solve initial value problem and unpack solution
        ivp_res = solve_ivp(f_ivp, (0, T), self.x, t_eval=t_eval)

        if ivp_res.status == 0:
            t = ivp_res.t
            x = ivp_res.y
        else:
            raise RuntimeError("scipy.integrate.solve_ivp failed")
        
        # Calculate corresponding outputs
        if self.output:
            npts = x.shape[1]
            y = np.zeros((self.p, npts))

            if self.g_input and run_with_input:
                for i in np.arange(npts):
                    y[:, i] = self.g(x[:, i], u(t[i]))
            
            else:
                for i in np.arange(npts):
                    y[:, i] = self.g(x[:, i])
            
            if self.measurement_noise:
                w_samples = self.w.sample(k=num)
                y = y + w_samples

            res = (t, x, y)
        else:
            res = (t, x)
        
        # Update state
        if update_state:
            self.x = x[:, -1]

        return res


#########
# NOISE #
#########

class _BaseNoise():
    def __init__(self):
        return

class StationaryGaussianWhiteNoise(_BaseNoise):
    def __init__(self, n=1, mu=0, cov=1, seed=None):
        # Input checking
        if not(seed is None):
            is_int(seed, "seed")
        
        is_strict_pos_int(n, "n")
        self.n = n

        if self.n == 1:
            is_num(mu, "mu")
            is_pos_num(cov, "cov")
            self.mu = mu
            self.cov = cov
        else:
            is_vec(mu, "mu")
            if mu.size != self.n:
                raise ValueError(f"'mu' should have size {self.n} but has size {mu.size} instead.")
            self.mu = mu

            is_square_mat(cov, "cov")
            if cov.shape[0] != self.n:
                raise ValueError(f"'cov' should have shape ({self.n}, {self.n}) but has shape {cov.shape} instead.")
            if not np.all(np.linalg.eigvals(cov) >= 0):
                raise ValueError(f"'cov' should be positive semi-definite.")
            self.cov = cov
        
        self.rng = np.random.default_rng(seed=seed)

        return
    
    def sample(self, k=1):
        # Check input
        is_strict_pos_int(k, "k")

        # Generate noise
        if self.n == 1:
            res = (self.rng.standard_normal(size=k) + self.mu) * np.sqrt(self.cov)
            if k == 1:
                res = res.reshape((1, ))
            else:
                res = res.reshape((1, -1))
        else:
            res = self.rng.multivariate_normal(self.mu, self.cov, size=k)
            if k == 1:
                res = res.squeeze()
            else:
                res = res.transpose()

        return res


############
# EXAMPLES #
############

# Constants
_sigma = 10
_rho = 28
_beta = 8/3

# Mathematical pendulum
def pendulum(x):
    return np.array([x[1], -np.sin(x[0])])

# Classic Lorenz system
def lorenz(x):
    x_lor, y, z = x[0], x[1], x[2]
    res = np.array([_sigma*(y-x_lor), x_lor*(_rho-z)-y, x_lor*y-_beta*z])
    return res

# Lorenz system where independent pendulum influences frequency of oscillations in the Lorenz system
def pendulum_lorenz(x):
    x_pen = pendulum(x[:2])
    x_lor = lorenz(x[2:])
    A = np.eye(3)*(1-x_pen[0])
    x_lor = np.matmul(A, x_lor)
    return np.concatenate((x_pen, x_lor))

# Turn exmaples into objects
_x0 = np.ones(2)
_x0[1] = 0
Pendulum = ContinuousNonlinearTimeInvariant(f=pendulum, n=2, x0=_x0)

_x0 = np.ones(3)
Lorenz = ContinuousNonlinearTimeInvariant(f=lorenz, n=3, x0=_x0)

_x0 = np.ones(5)
_x0[1] = 0
PendulumLorenz = ContinuousNonlinearTimeInvariant(f=pendulum_lorenz, n=5, x0=_x0)