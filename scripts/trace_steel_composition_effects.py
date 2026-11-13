"""Trace analytically defined composition descriptors back to recorded element differences."""
import numpy as np
from matminer.utils.data import MagpieData
from pymatgen.core import Element


def ratio_parts(a, b, da, db, side):
    """Return endpoint quotient multipliers for numerator and denominator changes."""
    if side == "low":
        return 1/db, -(a/da)/db
    if side == "high":
        return 1/da, -(b/db)/da
    average_den = .5*(da+db)
    return 1/average_den, -.5*(a/da + b/db)/average_den


def power_slope(a,b,p):
    # Algebraic secant of x**p with the exact derivative at equal endpoints.
    return sum(a**(p-1-k)*b**k for k in range(p))


class CompositionPaths:
    def __init__(self, archive):
        self.names = list(archive["columns"])
        self.ids = list(archive["sample_ids"])
        self.x = archive["X"]
        idx = [k for k,n in enumerate(self.names) if n.startswith("ElementFraction|") and self.x[:,k].any()]
        self.e = [self.names[k].split("|")[1] for k in idx]
        self.c = self.x[:,idx]
        self.fe = self.e.index("Fe")
        self.driver = [i for i in range(len(idx)) if i!=self.fe]
        self.labels = [self.e[i] for i in self.driver]
        atoms = [Element(e) for e in self.e]
        self.mass = np.array([float(e.atomic_mass) for e in atoms])
        mass = self.c*self.mass
        self.w = mass/mass.sum(axis=1,keepdims=True)
        np.testing.assert_allclose(self.c.sum(axis=1),1.,atol=1e-12,rtol=0)
        magpie = MagpieData()
        props = {n.split()[-1] for n in self.names if n.startswith("ElementProperty|")}
        self.p = {n:np.asarray(magpie.get_elemental_properties(atoms,n),dtype=float)
                  for n in props | {"MiracleRadius"}}
        self.specs = {}
        self.rebuild_error = 0.
        for j,name in enumerate(self.names):
            spec = self.definition(name)
            if spec is None:
                continue
            kind, prop = spec
            calc = self.values(kind,prop)
            np.testing.assert_allclose(calc,self.x[:,j],atol=1e-8,rtol=1e-10,err_msg=name)
            self.rebuild_error = max(self.rebuild_error,float(np.max(abs(calc-self.x[:,j]))))
            self.specs[j] = spec

    def definition(self,name):
        family, _, label = name.partition("|")
        if family == "ElementFraction" and label in self.e:
            return "fraction", self.e.index(label)
        if family == "ElementProperty":
            _, stat, prop = label.split()
            if stat in ("mean","avg_dev"):
                return stat, prop
        if family == "ValenceOrbital":
            stat, orb = label.split()[:2]
            return ("mean" if stat=="avg" else "valence_fraction"), "N"+orb+"Valence"
        if family == "Stoichiometry" and label!="0-norm":
            return "norm",int(label.split("-")[0])
        if family == "YangSolidSolution" and label=="Yang delta":
            return "size","MiracleRadius"
        return None

    def values(self,kind,prop):
        c = self.c
        if kind=="fraction":
            return c[:,prop]
        if kind=="norm":
            return (c**prop).sum(axis=1)**(1/prop)
        p = self.p[prop]
        mean = c @ p
        if kind=="mean":
            return mean
        if kind=="avg_dev":
            return (c*np.abs(p-mean[:,None])).sum(axis=1)
        if kind=="valence_fraction":
            return mean/(c@self.p["NValence"])
        assert kind=="size"
        return np.sqrt((c*(1-p/mean[:,None])**2).sum(axis=1))

    def atomic_changes(self,i,j,coordinate,side):
        """Columns correspond to the real non-Fe composition differences, with Fe as balance."""
        d = self.driver
        change = np.zeros((len(self.e),len(d)))
        if coordinate == "atomic":
            delta = self.c[j,d]-self.c[i,d]
            change[d,np.arange(len(d))] = delta
            change[self.fe] = -delta
        else:
            assert coordinate=="weight"
            delta = self.w[j,d]-self.w[i,d]
            change[d,np.arange(len(d))] = delta/self.mass[d]
            change[self.fe] = -delta/self.mass[self.fe]
            n0, n1 = self.w[i]/self.mass,self.w[j]/self.mass
            k0,k1 = ratio_parts(n0,n1,n0.sum(),n1.sum(),side)
            change = k0*change + k1[:,None]*change.sum(axis=0)
        np.testing.assert_allclose(change.sum(axis=1),self.c[j]-self.c[i],atol=2e-14,rtol=0)
        return change

    def coefficients(self,kind,prop,a,b,side):
        if kind=="fraction":
            return np.eye(len(a))[prop]
        if kind=="norm":
            za,zb = np.linalg.norm(a,ord=prop),np.linalg.norm(b,ord=prop)
            return power_slope(a,b,prop)/power_slope(za,zb,prop)
        p = self.p[prop]
        ma,mb = a@p,b@p
        if kind=="mean":
            return p
        if kind=="avg_dev":
            va,vb = abs(p-ma),abs(p-mb)
            if abs(mb-ma)>1e-12*max(1,abs(ma),abs(mb)):
                slope = (vb-va)/(mb-ma)
            else:
                slope = np.sign(.5*(ma+mb)-p)
            if side=="low":
                c,v = b,va
            elif side=="high":
                c,v = a,vb
            else:
                c,v = .5*(a+b),.5*(va+vb)
            return v + (c@slope)*p
        if kind=="valence_fraction":
            valence = self.p["NValence"]
            k0,k1 = ratio_parts(ma,mb,a@valence,b@valence,side)
            return k0*p+k1*valence
        assert kind=="size"
        # std(radius)/mean(radius). Centering at Fe avoids subtracting large squared radii.
        r = p-p[self.fe]
        ra,rb = a@r,b@r
        sa,sb = np.sqrt(a@((r-ra)**2)),np.sqrt(b@((r-rb)**2))
        if sa+sb==0:
            return np.zeros_like(r)
        std = (r*r-(ra+rb)*r)/(sa+sb)
        k0,k1 = ratio_parts(sa,sb,ma,mb,side)
        return k0*std+k1*p

    def shares(self,i,j,coordinate="weight",side="symmetric"):
        """Return signed fractions of each descriptor change; leave unsupported inputs unallocated."""
        dc = self.atomic_changes(i,j,coordinate,side)
        a,b = self.c[[i,j]]
        matrix = np.zeros((len(self.names),len(self.labels)))
        valid = np.zeros(len(self.names),dtype=bool)
        for k,(kind,prop) in self.specs.items():
            v = self.coefficients(kind,prop,a,b,side) @ dc
            xa,xb = self.x[[i,j],k]
            delta = xb-xa
            np.testing.assert_allclose(v.sum(),delta,atol=2e-9,rtol=1e-8,err_msg=self.names[k])
            if abs(delta)>1e-10*max(1.,abs(xa),abs(xb)):
                # Normalize by the analytically reconstructed change after checking the archive.
                matrix[k] = v/v.sum()
                valid[k] = True
        return matrix,valid
