from sympy.stats import *
from nbag import construct
import sympy.stats


def arcsin(a=0, b=1, name=None):
    return construct(sympy.stats.Arcsin, name, a, b)

def benini(alpha, beta, sigma, name=None):
    return construct(sympy.stats.Benini, name, alpha, beta, sigma)

def bernoulli(p, succ=1, fail=0, name=None):
    return construct(sympy.stats.Bernoulli, name, p, succ, fail)

def beta(alpha, beta, name=None):
    return construct(sympy.stats.Beta, name, alpha, beta)

def betaBinomial(n, alpha, beta, name=None):
    return construct(sympy.stats.BetaBinomial, name, n, alpha, beta)

def betaNoncentral(alpha, beta, lamda, name=None):
    return construct(sympy.stats.BetaNoncentral, name, alpha, beta, lamda)

def betaPrime(alpha, beta, name=None):
    return construct(sympy.stats.BetaPrime, name, alpha, beta)

def binomial(n, p, succ=1, fail=0, name=None):
    return construct(sympy.stats.Binomial, name, n, p, succ, fail)

def boundedPareto(alpha, left, right, name=None):
    return construct(sympy.stats.BoundedPareto, name, alpha, left, right)

def cauchy(x0, gamma, name=None):
    return construct(sympy.stats.Cauchy, name, x0, gamma)

def chi(k, name=None):
    return construct(sympy.stats.Chi, name, k)

def chiNoncentral(k, l, name=None):
    return construct(sympy.stats.ChiNoncentral, name, k, l)

def chiSquared(k, name=None):
    return construct(sympy.stats.ChiSquared, name, k)

def coin(p=1/2, name=None):
    return construct(sympy.stats.Coin, name, p)

def dagum(p, a, b, name=None):
    return construct(sympy.stats.Dagum, name, p, a, b)

def die(sides=6, name=None):
    return construct(sympy.stats.Die, name, sides)

def discreteUniform(items, name=None):
    return construct(sympy.stats.DiscreteUniform, name, items)

def erlang(k, l, name=None):
    return construct(sympy.stats.Erlang, name, k, l)

def exGaussian(mean, std, rate, name=None):
    return construct(sympy.stats.ExGaussian, name, mean, std, rate)

def exponential(rate, name=None):
    return construct(sympy.stats.Exponential, name, rate)

def exponentialPower(mu, alpha, beta, name=None):
    return construct(sympy.stats.ExponentialPower, name, mu, alpha, beta)

def fDistribution(d1, d2, name=None):
    return construct(sympy.stats.FDistribution, name, d1, d2)

def finiteRV(density, name=None, **kwargs):
    return construct(sympy.stats.FiniteRV, name, density, **kwargs)

def fisherZ(d1, d2, name=None):
    return construct(sympy.stats.FisherZ, name, d1, d2)

def florySchulz(a, name=None):
    return construct(sympy.stats.FlorySchulz, name, a)

def frechet(a, s=1, m=0, name=None):
    return construct(sympy.stats.Frechet, name, a, s, m)

def gamma(k, theta, name=None):
    return construct(sympy.stats.Gamma, name, k, theta)

def gammaInverse(a, b, name=None):
    return construct(sympy.stats.GammaInverse, name, a, b)

def gaussianInverse(mean, shape, name=None):
    return construct(sympy.stats.GaussianInverse, name, mean, shape)

def geometric(p, name=None):
    return construct(sympy.stats.Geometric, name, p)

def gompertz(b, eta, name=None):
    return construct(sympy.stats.Gompertz, name, b, eta)

def gumbel(beta, mu, minimum=False, name=None):
    return construct(sympy.stats.Gumbel, name, beta, mu, minimum)

def hermite(a1, a2, name=None):
    return construct(sympy.stats.Hermite, name, a1, a2)

def hypergeometric(N, m, n, name=None):
    return construct(sympy.stats.Hypergeometric, name, N, m, n)

def idealSoliton(k, name=None):
    return construct(sympy.stats.IdealSoliton, name, k)

def kumaraswamy(a, b, name=None):
    return construct(sympy.stats.Kumaraswamy, name, a, b)

def laplace(mu, b, name=None):
    return construct(sympy.stats.Laplace, name, mu, b)

def levy(mu, c, name=None):
    return construct(sympy.stats.Levy, name, mu, c)

def logCauchy(mu, sigma, name=None):
    return construct(sympy.stats.LogCauchy, name, mu, sigma)

def logLogistic(alpha, beta, name=None):
    return construct(sympy.stats.LogLogistic, name, alpha, beta)

def logNormal(mean, std, name=None):
    return construct(sympy.stats.LogNormal, name, mean, std)

def logarithmic(p, name=None):
    return construct(sympy.stats.Logarithmic, name, p)

def logistic(mu, s, name=None):
    return construct(sympy.stats.Logistic, name, mu, s)

def logitNormal(mu, s, name=None):
    return construct(sympy.stats.LogitNormal, name, mu, s)

def lomax(alpha, lamda, name=None):
    return construct(sympy.stats.Lomax, name, alpha, lamda)

def maxwell(a, name=None):
    return construct(sympy.stats.Maxwell, name, a)

def moyal(mu, sigma, name=None):
    return construct(sympy.stats.Moyal, name, mu, sigma)

def multivariateLaplace(mu, sigma, name=None):
    return construct(sympy.stats.MultivariateLaplace, name, mu, sigma)

def multivariateNormal(mu, sigma, name=None):
    return construct(sympy.stats.MultivariateNormal, name, mu, sigma)

def nakagami(mu, omega, name=None):
    return construct(sympy.stats.Nakagami, name, mu, omega)

def negativeBinomial(r, p, name=None):
    return construct(sympy.stats.NegativeBinomial, name, r, p)

def normal(mean, std, name=None):
    return construct(sympy.stats.Normal, name, mean, std)

def pareto(xm, alpha, name=None):
    return construct(sympy.stats.Pareto, name, xm, alpha)

def poisson(lamda, name=None):
    return construct(sympy.stats.Poisson, name, lamda)

def powerFunction(alpha, a, b, name=None):
    return construct(sympy.stats.PowerFunction, name, alpha, a, b)

def quadraticU(a, b, name=None):
    return construct(sympy.stats.QuadraticU, name, a, b)

def rademacher(name=None):
    return construct(sympy.stats.Rademacher, name)

def raisedCosine(mu, s, name=None):
    return construct(sympy.stats.RaisedCosine, name, mu, s)

def rayleigh(sigma, name=None):
    return construct(sympy.stats.Rayleigh, name, sigma)

def reciprocal(a, b, name=None):
    return construct(sympy.stats.Reciprocal, name, a, b)

def robustSoliton(k, delta, c, name=None):
    return construct(sympy.stats.RobustSoliton, name, k, delta, c)

def shiftedGompertz(b, eta, name=None):
    return construct(sympy.stats.ShiftedGompertz, name, b, eta)

def skellam(mu1, mu2, name=None):
    return construct(sympy.stats.Skellam, name, mu1, mu2)

def studentT(nu, name=None):
    return construct(sympy.stats.StudentT, name, nu)

def trapezoidal(a, b, c, d, name=None):
    return construct(sympy.stats.Trapezoidal, name, a, b, c, d)

def triangular(a, b, c, name=None):
    return construct(sympy.stats.Triangular, name, a, b, c)

def uniform(left, right, name=None):
    return construct(sympy.stats.Uniform, name, left, right)

def uniformSum(n, name=None):
    return construct(sympy.stats.UniformSum, name, n)

def vonMises(mu, k, name=None):
    return construct(sympy.stats.VonMises, name, mu, k)

def gaussianInverse(mean, shape, name=None):
    return construct(sympy.stats.GaussianInverse, name, mean, shape)

def weibull(alpha, beta, name=None):
    return construct(sympy.stats.Weibull, name, alpha, beta)

def wignerSemicircle(R, name=None):
    return construct(sympy.stats.WignerSemicircle, name, R)

def yuleSimon(rho, name=None):
    return construct(sympy.stats.YuleSimon, name, rho)

def zeta(s, name=None):
    return construct(sympy.stats.Zeta, name, s)

