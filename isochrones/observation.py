"""Observation tree classes for stellar parameter fitting.

This module provides tree-based data structures for representing stellar
observations and their hierarchical relationships, including photometric
measurements, spectroscopic data, and model evaluation.

Classes:
    Node: Base tree node class.
    ModelNode: Node representing a model with evaluation capabilities.
    Source: Leaf node representing an observational source.
    Observation: Node representing an observation.
    ObservationTree: Hierarchical tree of observations and sources.
"""

from __future__ import print_function, division

import os
import re
import sys
from typing import TYPE_CHECKING, Optional, Any, Dict, List, Union, Tuple

from .config import on_rtd
from .logger import getLogger

if not on_rtd:
    import numpy as np
    import pandas as pd
    from configobj import ConfigObj

    from asciitree import LeftAligned, Traversal
    from asciitree.drawing import BoxStyle, BOX_DOUBLE, BOX_BLANK

    from collections import OrderedDict

    from itertools import chain, count

    LOG_ONE_OVER_ROOT_2PI = np.log(1.0 / np.sqrt(2 * np.pi))

imap = map
izip = zip

from .isochrone import get_ichrone
from .utils import addmags, distance, fast_addmags


class NodeTraversal(Traversal):
    """Custom subclass to traverse tree for ASCII printing.

    This traversal class is used to render observation trees as ASCII
    diagrams, optionally showing model values and likelihoods.
    """

    def __init__(self, pars: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Initialize the node traversal.

        Args:
            pars: Optional dictionary of parameter values for displaying
                model evaluations.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        self.pars = pars
        super(NodeTraversal, self).__init__(**kwargs)

    def get_children(self, node: "Node") -> List["Node"]:
        """Get children of a node."""
        return node.children

    def get_root(self, node: "Node") -> "Node":
        """Get the root node."""
        return node

    def get_text(self, node: "Node") -> str:
        """Get the text representation of a node.

        Args:
            node: The node to get text for.

        Returns:
            str: ASCII representation of the node.
        """
        text = node.label
        if self.pars is not None:
            if hasattr(node, "model_mag"):
                text += "; model={:.2f} ({})".format(
                    node.model_mag(self.pars), node.lnlike(self.pars)  # type: ignore[attr-defined]
                )
            if type(node) == ModelNode:
                root = node.get_root()
                if hasattr(root, "spectroscopy"):
                    if node.label in root.spectroscopy:
                        for k, v in root.spectroscopy[node.label].items():
                            text += ", {}={}".format(k, v)

                            modval = node.evaluate(self.pars[node.label], k)
                            lnl = -0.5 * (modval - v[0]) ** 2 / v[1] ** 2
                            text += "; model={} ({})".format(modval, lnl)
                    if node.label in root.limits:
                        for k, v in root.limits[node.label].items():
                            text += ", {} limits={}".format(k, v)
                if hasattr(root, "parallax"):
                    if node.index in root.parallax:
                        plx, u_plx = root.parallax[node.index]
                        text += ", parallax={}".format((plx, u_plx))
                        modval = node.evaluate(self.pars[node.label], "parallax")
                        lnl = -0.5 * (modval - plx) ** 2 / u_plx ** 2
                        text += "; model={} ({})".format(modval, lnl)
                if hasattr(root, "AV"):
                    if node.index in root.AV:
                        AV, u_AV = root.AV[node.index]
                        text += ", AV={}".format((AV, u_AV))
                        modval = node.evaluate(self.pars[node.label], "AV")
                        lnl = -0.5 * (modval - plx) ** 2 / u_AV ** 2
                        text += "; model={} ({})".format(modval, lnl)

                text += ": {}".format(self.pars[node.label])

        else:
            if type(node) == ModelNode:
                root = node.get_root()
                if hasattr(root, "spectroscopy"):
                    if node.label in root.spectroscopy:
                        for k, v in root.spectroscopy[node.label].items():
                            text += ", {}={}".format(k, v)
                    if node.index in root.parallax:
                        text += ", parallax={}".format(root.parallax[node.index])
                    if node.index in root.AV:
                        text += ", AV={}".format(root.AV[node.index])
                    if node.label in root.limits:
                        for k, v in root.limits[node.label].items():
                            text += ", {} limits={}".format(k, v)
        return text


class MyLeftAligned(LeftAligned):
    """For custom ASCII tree printing.

    This class renders observation trees as ASCII diagrams with
    optional parameter display.
    """

    pars: Optional[Dict[str, Any]] = None

    def __init__(self, pars: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """Initialize the left-aligned tree renderer.

        Args:
            pars: Optional dictionary of parameter values for displaying
                model evaluations.
            **kwargs: Additional keyword arguments passed to parent class.
        """
        self.pars = pars
        self.traverse = NodeTraversal(pars)
        super(MyLeftAligned, self).__init__(**kwargs)


class Node:
    """Base tree node class for observation hierarchy.

    This class provides the basic tree structure for representing
    hierarchical relationships between observations, sources, and models.

    Attributes:
        label: The name/identifier for this node.
        parent: The parent node, or None if this is the root.
        children: List of child nodes.
    """

    def __init__(self, label: str) -> None:
        """Initialize a Node.

        Args:
            label: The name/identifier for this node.
        """
        self._label: str = label
        self.parent: Optional["Node"] = None
        self.children: List["Node"] = []
        self._leaves: Optional[List["Node"]] = None
        self.limits: Dict[str, Dict[str, Tuple[float, float]]] = {}
        self.parallax: Dict[int, Tuple[float, float]] = {}
        self.AV: Dict[int, Tuple[float, float]] = {}

    @property
    def label(self) -> str:
        return self._label

    def __iter__(self):
        """Iterate through tree, leaves first."""
        for node in chain(*imap(iter, self.children)):
            yield node
        yield self

    def __getitem__(self, ind: int) -> "Node":
        """Get node by index.

        Args:
            ind: Index of the node in the tree iteration order.

        Returns:
            The node at the specified index.
        """
        for n, i in izip(self, count()):
            if i == ind:
                return n
        raise IndexError("Node index out of range")

    @property
    def is_root(self) -> bool:
        """Check if this node is the root of the tree."""
        return self.parent is None

    def get_root(self) -> "Node":
        """Get the root node of the tree.

        Returns:
            The root node.
        """
        if self.is_root:
            return self
        else:
            if self.parent:
                return self.parent.get_root()
            return self

    def get_ancestors(self) -> List["Node"]:
        """Get all ancestor nodes from parent to root.

        Returns:
            List of ancestor nodes, excluding the root.
        """
        if self.parent is None or self.parent.is_root:
            return []
        else:
            return [self.parent] + self.parent.get_ancestors()  # type: ignore[return-value]

    def print_ascii(self, fout: Optional[Any] = None, pars: Optional[Dict[str, Any]] = None) -> None:
        """Print the tree as an ASCII diagram.

        Args:
            fout: Optional file-like object to write to. If None, prints to stdout.
            pars: Optional dictionary of parameter values for displaying
                model evaluations.
        """
        box_tr = MyLeftAligned(pars, draw=BoxStyle(gfx=BOX_DOUBLE, horiz_len=1))
        if fout is None:
            print(box_tr(self))
        else:
            fout.write(box_tr(self))

    @property
    def is_leaf(self) -> bool:
        """Check if this node is a leaf (has no children and is not the root)."""
        return len(self.children) == 0 and not self.is_root

    def _clear_leaves(self) -> None:
        """Clear the cached leaves list."""
        self._leaves = None

    def _clear_all_leaves(self) -> None:
        """Clear leaves cache for this node and all ancestors."""
        if self.parent is not None:
            self.parent._clear_all_leaves()
        self._clear_leaves()

    def add_child(self, node: "Node") -> None:
        """Add a child node.

        Args:
            node: The node to add as a child.
        """
        node.parent = self
        self.children.append(node)
        self._clear_all_leaves()

    def remove_children(self) -> None:
        """Remove all child nodes."""
        self.children = []
        self._clear_all_leaves()

    def remove_child(self, label: str) -> None:
        """Remove a child node by label."""
        ind = None
        for i, c in enumerate(self.children):
            if c.label == label:
                ind = i

        if ind is None:
            getLogger().warning("No child labeled {}.".format(label))
            return
        self.children.pop(ind)
        self._clear_all_leaves()

    def attach_to_parent(self, node):
        # detach from current parent, if necessary
        if self.parent is not None:
            self.parent.remove_child(self.label)

        node.children += [self]
        self.parent = node
        self._clear_all_leaves()

    @property
    def leaves(self):
        if self._leaves is None:
            self._leaves = self._get_leaves()
        return self._leaves

    def _get_leaves(self):
        if self.is_leaf:
            return [self]
        else:
            leaves = []
            for c in self.children:
                leaves += c._get_leaves()
            return leaves

    def select_leaves(self, name):
        """Returns all leaves under all nodes matching name

        """

        if self.is_leaf:
            return [self] if re.search(name, self.label) else []
        else:
            leaves = []
            if re.search(name, self.label):
                for c in self.children:
                    leaves += c._get_leaves()  # all leaves
            else:
                for c in self.children:
                    leaves += c.select_leaves(name)  # only matching ones
            return leaves

    @property
    def leaf_labels(self):
        return [l.label for l in self.leaves]

    def get_leaf(self, label):
        for l in self.leaves:
            if label == l.label:
                return l

    def get_obs_nodes(self):
        return [l for l in self if isinstance(l, ObsNode)]

    @property
    def obs_leaf_nodes(self):
        return self.get_obs_leaves()

    def get_obs_leaves(self):
        """Returns the last obs nodes that are leaves
        """
        obs_leaves = []
        for n in self:
            if n.is_leaf:
                if isinstance(n, ModelNode):
                    l = n.parent
                else:
                    l = n
            if l not in obs_leaves:
                obs_leaves.append(l)
        return obs_leaves

    def get_model_nodes(self):
        return [l for l in self._get_leaves() if isinstance(l, ModelNode)]

    @property
    def N_model_nodes(self):
        return len(self.get_model_nodes())

    def print_tree(self):
        print(self.label)

    def __str__(self):
        return self.label

    def __repr__(self):
        if self.is_leaf:
            s = "<{} '{}', parent='{}'>".format(self.__class__, self.label, self.parent)
        else:
            child_labels = [str(c) for c in self.children]
            s = "<{} '{}', parent='{}', children={}>".format(
                self.__class__, self.label, self.parent, child_labels
            )
        return s


class ObsNode(Node):
    """Observation node in the observation tree.

    This class represents an observation (photometric measurement) in the
    hierarchical observation tree. It connects a specific observation and
    source to potential model nodes for stellar parameter fitting.

    Attributes:
        observation: The Observation object containing observation metadata.
        source: The Source object containing the measurement data.
        reference: Optional reference ObsNode for relative photometry.
    """

    def __init__(
        self,
        observation: "Observation",
        source: "Source",
        ref_node: Optional["ObsNode"] = None
    ) -> None:
        """Initialize an ObsNode.

        Args:
            observation: The Observation object with metadata.
            source: The Source object with measurement data.
            ref_node: Optional reference node for differential photometry.
        """
        self.observation: Optional["Observation"] = observation
        self.source: Optional["Source"] = source
        self.reference: Optional["ObsNode"] = ref_node

        self.children: List["Node"] = []
        self.parent: Optional["Node"] = None
        self._leaves: Optional[List["Node"]] = None

        self._inds: Optional[List[int]] = None
        self._n_params: Optional[int] = None
        self._Nstars: Optional[Dict[int, int]] = None

        self._cache_key: Optional[tuple] = None
        self._cache_val: Optional[float] = None

    @property
    def instrument(self) -> str:
        """The instrument name for this observation."""
        assert self.observation is not None
        return self.observation.name

    @property
    def band(self) -> str:
        """The photometric band for this observation."""
        assert self.observation is not None
        return self.observation.band

    @property
    def value(self) -> Tuple[Any, Any]:
        """The (magnitude, error) tuple for this observation."""
        assert self.source is not None
        return (self.source.mag, self.source.e_mag)

    @property
    def resolution(self) -> float:
        """The spatial resolution of the observation."""
        assert self.observation is not None
        return self.observation.resolution

    @property
    def relative(self) -> bool:
        """Whether this is a relative/differential measurement."""
        assert self.source is not None
        return self.source.relative

    @property
    def separation(self) -> float:
        """Angular separation from the reference position."""
        assert self.source is not None
        return self.source.separation

    @property
    def pa(self) -> float:
        """Position angle from the reference position."""
        assert self.source is not None
        return self.source.pa

    @property
    def value_str(self) -> str:
        """String representation of the magnitude and error."""
        return "({:.2f}, {:.2f})".format(*self.value)

    def distance(self, other: "ObsNode") -> float:
        """Calculate coordinate distance from another ObsNode.

        Args:
            other: Another ObsNode to calculate distance to.

        Returns:
            float: Angular separation between the two observations.
        """
        return distance((self.separation, self.pa), (other.separation, other.pa))

    def _in_same_observation(self, other: "ObsNode") -> bool:
        """Check if two ObsNodes are from the same instrument and band."""
        return self.instrument == other.instrument and self.band == other.band

    @property
    def n_params(self) -> int:
        """Number of free parameters for this observation."""
        if self._n_params is None:
            self._n_params = 5 * len(self.leaves)
        return self._n_params

    def _get_inds(self) -> List[int]:
        """Get unique sorted indices of model systems."""
        inds = [n.index for n in self.leaves]
        inds = sorted(list(set(inds)))
        return inds

    def _clear_leaves(self) -> None:
        """Clear cached values when leaves change."""
        self._leaves = None
        self._inds = None
        self._n_params = None
        self._Nstars = None

    @property
    def Nstars(self) -> Dict[int, int]:
        """Dictionary of number of stars per system."""
        if self._Nstars is None:
            N: Dict[int, int] = {}
            for n in self.get_model_nodes():
                if n.index not in N:
                    N[n.index] = 1
                else:
                    N[n.index] += 1
            self._Nstars = N
        return self._Nstars

    @property
    def systems(self) -> List[int]:
        """List of unique system indices."""
        lst = sorted(self.Nstars.keys())
        return lst

    @property
    def inds(self) -> List[int]:
        """Unique sorted indices of model systems."""
        if self._inds is None:
            self._inds = self._get_inds()
        return self._inds

    @property
    def label(self) -> str:
        """String label for this observation node."""
        assert self.source is not None
        if self.source.relative:
            band_str = "delta-{}".format(self.band)
        else:
            band_str = self.band
        return "{} {}={} @({:.2f}, {:.0f} [{:.2f}])".format(
            self.instrument, band_str, self.value_str, self.separation, self.pa, self.resolution
        )

    @property
    def obsname(self):
        return "{}-{}".format(self.instrument, self.band)

    def get_system(self, ind):
        system = []
        for l in self.get_root().leaves:
            try:
                if l.index == ind:
                    system.append(l)
            except AttributeError:
                pass
        return system

    def add_model(self, ic, N=1, index=0):
        """
        Should only be able to do this to a leaf node.

        Either N and index both integers OR index is
        list of length=N
        """
        if type(index) in [list, tuple]:
            if len(index) != N:
                raise ValueError("If a list, index must be of length N.")
        else:
            index = [index] * N

        for idx in index:
            existing = self.get_system(idx)
            tag = len(existing)
            self.add_child(ModelNode(ic, index=idx, tag=tag))

    def model_mag(self, model_values, use_cache=True):
        """
        pardict is a dictionary of parameters for all leaves
        gets converted back to traditional parameter vector
        """
        # if pardict == self._cache_key and use_cache:
        #     #print('{}: using cached'.format(self))
        #     return self._cache_val

        # #print('{}: calculating'.format(self))
        # self._cache_key = pardict

        return addmags(*[model_values[n.label][self.band] for n in self.leaves])

    def lnlike(self, model_values, use_cache=True):
        """
        returns log-likelihood of this observation

        pardict is a dictionary of parameters for all leaves
        gets converted back to traditional parameter vector
        """

        mag, dmag = self.value
        if np.isnan(dmag):
            return 0
        if self.relative:
            # If this *is* the reference, just return
            if self.reference is None:
                return 0
            mod = self.model_mag(model_values, use_cache=use_cache) - self.reference.model_mag(
                model_values, use_cache=use_cache
            )
            mag -= self.reference.value[0]
        else:
            mod = self.model_mag(model_values, use_cache=use_cache)

        lnl = -0.5 * (mag - mod) ** 2 / dmag ** 2 + LOG_ONE_OVER_ROOT_2PI + np.log(dmag)

        # getLogger().debug('{} {}: mag={}, mod={}, lnlike={}'.format(self.instrument,
        #                                                         self.band,
        #                                                         mag,mod,lnl))
        return lnl


class DummyObsNode(ObsNode):
    """Dummy observation node for placeholder/inactive observations.

    This class is used as a placeholder when an observation node needs
    to be temporarily disabled or when no observation is available.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a DummyObsNode."""
        self.observation: Optional["Observation"] = None
        self.source: Optional["Source"] = None
        self.reference: Optional["ObsNode"] = None

        self.children: List["Node"] = []
        self.parent: Optional["Node"] = None
        self._leaves: Optional[List["Node"]] = None

        self._inds: Optional[List[int]] = None
        self._n_params: Optional[int] = None
        self._Nstars: Optional[Dict[int, int]] = None

        self._cache_key: Optional[tuple] = None
        self._cache_val: Optional[float] = None

    @property
    def label(self) -> str:
        """String label for dummy node."""
        return "[dummy]"

    @property
    def value(self) -> Tuple[None, None]:
        """Value returns None for dummy node."""
        return None, None

    def lnlike(self, *args: Any, **kwargs: Any) -> float:
        """Log-likelihood is always zero for dummy node."""
        return 0


class ModelNode(Node):
    """Model node representing a stellar model in the observation tree.

    ModelNodes are always leaf nodes that connect to isochrone models
    for stellar parameter evaluation. The index keeps track of which
    physical system (star) the node belongs to.

    Attributes:
        index: Index of the physical system this model represents.
        tag: Additional tag for distinguishing multiple models in same system.
    """

    def __init__(self, ic: Any, index: int = 0, tag: int = 0) -> None:
        """Initialize a ModelNode.

        Args:
            ic: The isochrone/interpolator object.
            index: Index of the physical system (star).
            tag: Additional tag for multiple models in same system.
        """
        self._ic = ic
        self.index = index
        self.tag = tag

        self.children: List["Node"] = []
        self.parent: Optional["Node"] = None
        self._leaves: Optional[List["Node"]] = None

    @property
    def label(self) -> str:
        """String label for this model node."""
        return "{}_{}".format(self.index, self.tag)

    @property
    def ic(self) -> Any:
        """The isochrone interpolator (lazy initialization)."""
        if type(self._ic) == type:
            self._ic = self._ic()
        return self._ic

    def get_obs_ancestors(self) -> List["ObsNode"]:
        """Get all observation ancestors in the tree."""
        nodes = self.get_ancestors()
        return [n for n in nodes if isinstance(n, ObsNode)]

    @property
    def contributing_observations(self) -> List[str]:
        """The instrument-band pairs for all observations feeding this model."""
        return [n.obsname for n in self.get_obs_ancestors()]

    def evaluate(self, p: Tuple[float, ...], prop: str) -> float:
        """Evaluate a stellar property from model parameters.

        Args:
            p: Tuple of (mass, age, feh) parameters.
            prop: Property name to evaluate ('mass', 'age', 'feh', 'Teff', etc.)

        Returns:
            float: The evaluated property value.

        Raises:
            ValueError: If the property cannot be evaluated.
        """
        if prop in self.ic.bands:
            return self.evaluate_mag(p, prop)
        elif prop == "mass":
            return p[0]
        elif prop == "age":
            return p[1]
        elif prop == "feh":
            return p[2]
        elif prop in ["Teff", "logg", "radius", "density"]:
            return getattr(self.ic, prop)(*p[:3])
        else:
            raise ValueError("property {} cannot be evaluated by Isochrone.".format(prop))

    def evaluate_mag(self, p: Tuple[float, ...], band: str) -> float:
        """Evaluate magnitude in a specific band."""
        return self.ic.mag[band](*p)

    def lnlike(self, *args: Any, **kwargs: Any) -> float:
        """Log-likelihood for model node (always zero, computed at ObsNode)."""
        return 0


class Source:
    """Represents an observational measurement (photometry).

    This class stores photometric measurement data including magnitude,
    uncertainty, and position information for differential photometry.

    Attributes:
        mag: The measured magnitude.
        e_mag: The magnitude uncertainty.
        separation: Angular separation from reference position.
        pa: Position angle from reference position.
        relative: Whether this is a relative/differential measurement.
        is_reference: Whether this is a reference source.
    """

    def __init__(
        self,
        mag: float,
        e_mag: float,
        separation: float = 0.0,
        pa: float = 0.0,
        relative: bool = False,
        is_reference: bool = False
    ) -> None:
        """Initialize a Source.

        Args:
            mag: Measured magnitude.
            e_mag: Magnitude uncertainty.
            separation: Angular separation from reference (degrees).
            pa: Position angle from reference (degrees).
            relative: Whether this is a relative measurement.
            is_reference: Whether this is the reference source.
        """
        self.mag = float(mag)
        self.e_mag = float(e_mag)
        self.separation = float(separation)
        self.pa = float(pa)
        self.relative = bool(relative)
        self.is_reference = bool(is_reference)

    def __str__(self) -> str:
        return "({}, {}) @({}, {})".format(self.mag, self.e_mag, self.separation, self.pa)

    def __repr__(self) -> str:
        return self.__str__()


class Star:
    """Theoretical counterpart of a Source.

    Represents a model star with parameters and position for comparison
    with observed Sources.
    """

    def __init__(self, pars: Dict[str, float], separation: float, pa: float) -> None:
        """Initialize a Star.

        Args:
            pars: Dictionary of stellar parameters.
            separation: Angular separation from reference.
            pa: Position angle from reference.
        """
        self.pars = pars
        self.separation = separation
        self.pa = pa

    def distance(self, other: "Star") -> float:
        """Calculate angular separation from another Star."""
        return distance((self.separation, self.pa), (other.separation, other.pa))


class Observation:
    """Contains relevant information about an imaging observation.

    Attributes:
        name: Identifying string (typically the instrument name).
        band: Photometric bandpass name.
    resolution: *approximate* angular resolution of instrument.
         used for source matching between observations
    sources: list of Source objects

    """

    def __init__(self, name, band, resolution, sources=None, relative=False):
        self.name = name
        self.band = band
        self.resolution = resolution
        if sources is not None:
            if not np.all(type(s) == Source for s in sources):
                raise ValueError("Source list must be all Source objects.")

        self.sources = []
        if sources is None:
            sources = []
        for s in sources:
            self.add_source(s)

        self.relative = relative
        self._set_reference()

    def observe(self, stars, unc, ic=None):
        """Creates and adds appropriate synthetic Source objects for list of stars (max 2 for now)
        """
        if ic is None:
            ic = get_ichrone("mist")

        if len(stars) > 2:
            raise NotImplementedError("No support yet for > 2 synthetic stars")

        mags = [ic(*s.pars)["{}_mag".format(self.band)].values[0] for s in stars]

        d = stars[0].distance(stars[1])

        if d < self.resolution:
            mag = addmags(*mags) + unc * np.random.randn()
            sources = [Source(mag, unc, stars[0].separation, stars[0].pa, relative=self.relative)]
        else:
            mags = np.array([m + unc * np.random.randn() for m in mags])
            if self.relative:
                mags -= mags.min()
            sources = [
                Source(m, unc, s.separation, s.pa, relative=self.relative) for m, s in zip(mags, stars)
            ]

        for s in sources:
            self.add_source(s)

        self._set_reference()

    def add_source(self, source):
        """
        Adds source to observation, keeping sorted order (in separation)
        """
        if not type(source) == Source:
            raise TypeError("Can only add Source object.")

        if len(self.sources) == 0:
            self.sources.append(source)
        else:
            ind = 0
            for s in self.sources:
                # Keep sorted order of separation
                if source.separation < s.separation:
                    break
                ind += 1

            self.sources.insert(ind, source)

        # self._set_reference()

    @property
    def brightest(self):
        mag0 = np.inf
        s0 = None
        for s in self.sources:
            if s.mag < mag0:
                mag0 = s.mag
                s0 = s
        return s0

    def _set_reference(self):
        """If relative, make sure reference node is set to brightest.
        """
        if len(self.sources) > 0:
            self.brightest.is_reference = True

    def __str__(self):
        return "{}-{}".format(self.name, self.band)

    def __repr__(self):
        return str(self)


class ObservationTree(Node):
    """Hierarchical tree structure for organizing stellar observations.

    Builds a tree of Nodes from a list of Observation objects. Organizes
    Observations from smallest to largest resolution, and at each stage
    attaches each source to the most probable match from the previous
    Observation. This allows modeling of multiple stars in a system with
    relative photometry.

    The tree structure allows matching observations across different
    instruments/bands while accounting for resolution differences.

    Attributes:
        spectroscopy: Dictionary of spectroscopic constraints (e.g., Teff, logg).
        limits: Dictionary of parameter limits.
        parallax: Dictionary of parallax measurements.
        AV: Dictionary of extinction values.
    """

    spec_props: List[str] = ["Teff", "logg", "feh", "density"]

    def __init__(
        self,
        observations: Optional[List["Observation"]] = None,
        name: Optional[str] = None
    ) -> None:
        """Initialize an ObservationTree.

        Args:
            observations: List of Observation objects to add to the tree.
            name: Optional name for the tree root.
        """
        if observations is None:
            observations = []

        if name is None:
            self.label: str = "root"
        else:
            self.label = name
        self.parent: Optional["Node"] = None

        self._observations: List["Observation"] = []
        self._build_tree()

        [self.add_observation(obs) for obs in observations]

        self._N: Optional[int] = None
        self._index: Optional[int] = None

        self.spectroscopy: Dict[str, Dict[str, Tuple[float, float]]] = {}

        self.limits: Dict[str, Dict[str, Tuple[float, float]]] = {}

        self.parallax: Dict[int, Tuple[float, float]] = {}

        self.AV: Dict[int, Tuple[float, float]] = {}

        self._Nstars: Optional[int] = None

        self._cache_key: Optional[tuple] = None
        self._cache_val: Optional[float] = None

    @property
    def name(self) -> str:
        """Name of the observation tree."""
        return self.label

    def _clear_cache(self) -> None:
        """Clear the likelihood cache."""
        self._cache_key = None
        self._cache_val = None

    @classmethod
    def from_df(cls, df: pd.DataFrame, **kwargs: Any) -> "ObservationTree":
        """Create an ObservationTree from a pandas DataFrame.

        Args:
            df: DataFrame with columns: name, band, resolution, mag, e_mag,
                separation, pa, relative.
            **kwargs: Additional arguments passed to the constructor.

        Returns:
            ObservationTree: The constructed observation tree.

        Raises:
            KeyError: If required columns are missing from DataFrame.
        """
        tree = cls(**kwargs)

        for (n, b), g in df.groupby(["name", "band"]):
            sources = [
                Source(**s[["mag", "e_mag", "separation", "pa", "relative"]]) for _, s in g.iterrows()
            ]
            obs = Observation(n, b, g.resolution.mean(), sources=sources, relative=g.relative.any())
            tree.add_observation(obs)

        return tree

    @classmethod
    def from_ini(cls, filename: str) -> "ObservationTree":
        """Create an ObservationTree from an INI config file.

        Args:
            filename: Path to the INI configuration file.

        Returns:
            ObservationTree: The constructed observation tree.
        """
        config = ConfigObj(filename)
        tree = cls()
        # TODO: implement parsing of config into observation tree
        return tree

    def to_df(self) -> pd.DataFrame:
        """Convert observations to a pandas DataFrame.

        Returns:
            DataFrame with columns: name, band, resolution, mag, e_mag,
            separation, pa, relative.
        """
        df = pd.DataFrame()
        name = []
        band = []
        resolution = []
        mag = []
        e_mag = []
        separation = []
        pa = []
        relative = []
        for o in self._observations:
            for s in o.sources:
                name.append(o.name)
                band.append(o.band)
                resolution.append(o.resolution)
                mag.append(s.mag)
                e_mag.append(s.e_mag)
                separation.append(s.separation)
                pa.append(s.pa)
                relative.append(s.relative)

        return pd.DataFrame(
            {
                "name": name,
                "band": band,
                "resolution": resolution,
                "mag": mag,
                "e_mag": e_mag,
                "separation": separation,
                "pa": pa,
                "relative": relative,
            }
        )

    def save_hdf(self, filename, path="", overwrite=False, append=False):
        """
        Writes all info necessary to recreate object to HDF file

        Saves table of photometry in DataFrame

        Saves model specification, spectroscopy, parallax to attrs
        """

        if os.path.exists(filename):
            store = pd.HDFStore(filename)
            if path in store:
                store.close()
                if overwrite:
                    os.remove(filename)
                elif not append:
                    raise IOError(
                        "{} in {} exists.  Set either overwrite or append option.".format(path, filename)
                    )
            else:
                store.close()

        df = self.to_df()
        df.to_hdf(filename, key=path + "/df", format="table")
        with pd.HDFStore(filename) as store:
            # store = pd.HDFStore(filename)
            attrs = store.get_storer(path + "/df").attrs
            attrs.spectroscopy = self.spectroscopy
            attrs.parallax = self.parallax
            attrs.AV = self.AV
            attrs.N = self._N
            attrs.index = self._index
            store.close()

    @classmethod
    def load_hdf(cls, filename, path="", ic=None):
        """
        Loads stored ObservationTree from file.

        You can provide the isochrone to use; or it will default to MIST

        TODO: saving and loading must be fixed!  save ic type, bands, etc.
        """
        store = pd.HDFStore(filename)
        try:
            samples = store[path + "/df"]
            attrs = store.get_storer(path + "/df").attrs
        except:
            store.close()
            raise
        df = store[path + "/df"]
        new = cls.from_df(df)

        if ic is None:
            ic = get_ichrone("mist")

        new.define_models(ic, N=attrs.N, index=attrs.index)
        new.spectroscopy = attrs.spectroscopy
        new.parallax = attrs.parallax
        new.AV = attrs.AV
        store.close()
        return new

    def add_observation(self, obs):
        """Adds an observation to observation list, keeping proper order
        """
        if len(self._observations) == 0:
            self._observations.append(obs)
        else:
            res = obs.resolution
            ind = 0
            for o in self._observations:
                if res > o.resolution:
                    break
                ind += 1
            self._observations.insert(ind, obs)

        self._build_tree()
        self._clear_cache()

    def add_spectroscopy(self, label="0_0", **props):
        """
        Adds spectroscopic measurement to particular star(s) (corresponding to individual model node)

        Default 0_0 should be primary star

        legal inputs are 'Teff', 'logg', 'feh', and in form (val, err)
        """
        if label not in self.leaf_labels:
            raise ValueError(
                "No model node named {} (must be in {}). Maybe define models first?".format(
                    label, self.leaf_labels
                )
            )
        for k, v in props.items():
            if k not in self.spec_props:
                raise ValueError("Illegal property {} (only {} allowed).".format(k, self.spec_props))
            if len(v) != 2:
                raise ValueError("Must provide (value, uncertainty) for {}.".format(k))

        if label not in self.spectroscopy:
            self.spectroscopy[label] = {}

        for k, v in props.items():
            self.spectroscopy[label][k] = v

        self._clear_cache()

    def add_limit(self, label="0_0", **props):
        """Define limits to spectroscopic property of particular stars.

        Usually will be used for 'logg', but 'Teff' and 'feh' will also work.

        In form (min, max): e.g., t.add_limit(logg=(3.0,None))

        None will be converted to (-)np.inf
        """

        if label not in self.leaf_labels:
            raise ValueError(
                "No model node named {} (must be in {}). Maybe define models first?".format(
                    label, self.leaf_labels
                )
            )
        for k, v in props.items():
            if k not in self.spec_props:
                raise ValueError("Illegal property {} (only {} allowed).".format(k, self.spec_props))
            if len(v) != 2:
                raise ValueError("Must provide (min, max) for {}. (`None` is allowed value)".format(k))

        if label not in self.limits:
            self.limits[label] = {}

        for k, v in props.items():
            vmin, vmax = v
            if vmin is None:
                vmin = -np.inf
            if vmax is None:
                vmax = np.inf
            self.limits[label][k] = (vmin, vmax)

        self._clear_cache()

    def add_parallax(self, plax, system=0):
        if len(plax) != 2:
            raise ValueError("Must enter (value,uncertainty).")
        if system not in self.systems:
            raise ValueError("{} not in systems ({}).".format(system, self.systems))

        self.parallax[system] = plax
        self._clear_cache()

    def add_AV(self, AV, system=0):
        if len(AV) != 2:
            raise ValueError("Must enter (value,uncertainty).")
        if system not in self.systems:
            raise ValueError("{} not in systems ({}).".format(system, self.systems))

        self.AV[system] = AV
        self._clear_cache()

    def define_models(self, ic, leaves=None, N=1, index=0):
        """
        N, index are either integers or lists of integers.

        N : number of model stars per observed star
        index : index of physical association

        leaves: either a list of leaves, or a pattern by which
        the leaves are selected (via `select_leaves`)

        If these are lists, then they are defined individually for
        each leaf.

        If `index` is a list, then each entry must be either
        an integer or a list of length `N` (where `N` is the corresponding
            entry in the `N` list.)

        This bugs up if you call it multiple times.  If you want
        to re-do a call to this function, please re-define the tree.
        """

        self.clear_models()

        if leaves is None:
            leaves = self._get_leaves()
        elif type(leaves) == type(""):
            leaves = self.select_leaves(leaves)

        # Sort leaves by distance, to ensure system 0 will be assigned
        # to the main reference star.

        if np.isscalar(N):
            N = np.ones(len(leaves)) * N
            # if np.size(index) > 1:
            #    index = [index]
        N = np.array(N).astype(int)

        if np.isscalar(index):
            index = np.ones_like(N) * index
        index = np.array(index).astype(int)

        # Add the appropriate number of model nodes to each
        #  star in the highest-resoluion image
        for s, n, i in zip(leaves, N, index):
            # Remove any previous model nodes (should do some checks here?)
            s.remove_children()
            s.add_model(ic, n, i)

        # For each system, make sure tag _0 is the brightest.
        self._fix_labels()

        self._N = N
        self._index = index

        self._clear_all_leaves()

    def _fix_labels(self):
        """For each system, make sure tag _0 is the brightest, and make sure
        system 0 contains the brightest star in the highest-resolution image
        """
        for s in self.systems:
            mag0 = np.inf
            n0 = None
            for n in self.get_system(s):
                if isinstance(n.parent, DummyObsNode):
                    continue
                mag, _ = n.parent.value
                if mag < mag0:
                    mag0 = mag
                    n0 = n

            # If brightest is not tag _0, then switch them.
            if n0 is not None and n0.tag != 0:
                n_other = self.get_leaf("{}_{}".format(s, 0))
                n_other.tag = n0.tag
                n0.tag = 0

    def get_system(self, ind):
        system = []
        for l in self.leaves:
            try:
                if l.index == ind:
                    system.append(l)
            except AttributeError:
                pass
        return system

    @property
    def observations(self):
        return self._observations

    def select_observations(self, name):
        """Returns nodes whose instrument-band matches 'name'
        """
        return [n for n in self.get_obs_nodes() if n.obsname == name]

    def clear_models(self):
        for n in self:
            if isinstance(n, ModelNode):
                n.parent.remove_child(n.label)

        self._clear_all_leaves()

    def trim(self):
        """
        Trims leaves from tree that are not observed at highest-resolution level

        This is a bit hacky-- what it does is
        """
        # Only allow leaves to stay on list (highest-resolution) level
        return

        for l in self._levels[-2::-1]:
            for n in l:
                if n.is_leaf:
                    n.parent.remove_child(n.label)

        self._clear_all_leaves()  # clears cached list of leaves

    def p2pardict(self, p):
        """
        Given leaf labels, turns parameter vector into pardict
        """
        d = {}
        N = self.Nstars
        i = 0
        for s in self.systems:
            age, feh, dist, AV = p[i + N[s] : i + N[s] + 4]
            for j in xrange(N[s]):
                l = "{}_{}".format(s, j)
                mass = p[i + j]
                d[l] = [mass, age, feh, dist, AV]
            i += N[s] + 4
        return d

    def pardict2p(self, pardict):
        """Convert from dictionary back to flat parameter vector
        """
        pars = []
        N = self.Nstars
        for s in self.systems:
            for i in range(N[s]):
                star = "{}_{}".format(s, i)
                pars.append(pardict[star][0])
            pars += pardict["{}_0".format(s)][1:]

        return pars

    @property
    def param_description(self):
        N = self.Nstars
        pars = []
        for s in self.systems:
            for j in xrange(N[s]):
                pars.append("eep_{}_{}".format(s, j))
            for p in ["age", "feh", "distance", "AV"]:
                pars.append("{}_{}".format(p, s))
        return pars

    @property
    def Nstars(self):
        if self._Nstars is None:
            N = {}
            for n in self.get_model_nodes():
                if n.index not in N:
                    N[n.index] = 1
                else:
                    N[n.index] += 1
            self._Nstars = N

        return self._Nstars

    @property
    def systems(self):
        # fix this! make sure it is unique!!!
        lst = list(chain(*[c.systems for c in self.children]))
        return sorted(set(lst))

    def print_ascii(self, fout=None, p=None):
        pardict = None
        if p is not None:
            pardict = self.p2pardict(p)
        super(ObservationTree, self).print_ascii(fout, pardict)

    def lnlike(self, p, model_values, use_cache=True):
        """
        takes parameter vector, constructs pardict, returns sum of lnlikes of non-leaf nodes
        """
        pardict = self.p2pardict(p) if type(p) is not dict else p

        # TODO: do we still want caching?
        # if use_cache and self._cache_key is not None and np.all(p==self._cache_key):
        #     return self._cache_val
        # self._cache_key = p

        # lnlike from photometry
        lnl = 0
        for n in self:
            if n is not self:
                lnl += n.lnlike(model_values, use_cache=use_cache)
            if not np.isfinite(lnl):
                self._cache_val = -np.inf
                return -np.inf

        # lnlike from spectroscopy
        for l in self.spectroscopy:
            for prop, (val, err) in self.spectroscopy[l].items():
                mod = model_values[l][prop]
                lnl += -0.5 * (val - mod) ** 2 / err ** 2 + LOG_ONE_OVER_ROOT_2PI + np.log(err)
            if not np.isfinite(lnl):
                self._cache_val = -np.inf
                return -np.inf

        # enforce limits
        for l in self.limits:
            for prop, (vmin, vmax) in self.limits[l].items():
                mod = model_values[l][prop]
                if mod < vmin or mod > vmax or not np.isfinite(mod):
                    self._cache_val = -np.inf
                    return -np.inf

        # lnlike from parallax
        for s, (val, err) in self.parallax.items():
            dist = pardict["{}_0".format(s)][3]
            mod = 1.0 / dist * 1000.0
            lnl += -0.5 * (val - mod) ** 2 / err ** 2 + LOG_ONE_OVER_ROOT_2PI + np.log(err)

        # lnlike from AV
        for s, (val, err) in self.AV.items():
            AV = pardict["{}_0".format(s)][4]
            lnl += -0.5 * (val - AV) ** 2 / err ** 2 + LOG_ONE_OVER_ROOT_2PI + np.log(err)

        if not np.isfinite(lnl):
            self._cache_val = -np.inf
            return -np.inf

        self._cache_val = lnl
        return lnl

    def _find_closest(self, n0):
        """returns the node in the tree that is closest to n0, but not
          in the same observation
        """
        dmin = np.inf
        nclose = None

        ds = []
        nodes = []

        ds.append(np.inf)
        nodes.append(self)

        for n in self:
            if n is n0:
                continue
            try:
                if n._in_same_observation(n0):
                    continue
                ds.append(n.distance(n0))
                nodes.append(n)
            except AttributeError:
                pass

        inds = np.argsort(ds)
        ds = [ds[i] for i in inds]
        nodes = [nodes[i] for i in inds]

        for d, n in zip(ds, nodes):
            try:
                if d < n.resolution or n.resolution == -1:
                    return n
            except AttributeError:
                pass

        # If nothing else works
        return self

    def _build_tree(self):
        # reset leaf cache, children
        self._clear_all_leaves()
        self.children = []

        for i, o in enumerate(self._observations):
            s0 = o.brightest
            ref_node = ObsNode(o, s0)
            for s in o.sources:
                if s.relative and not s.is_reference:
                    node = ObsNode(o, s, ref_node=ref_node)
                elif s.relative and s.is_reference:
                    node = ref_node
                else:
                    node = ObsNode(o, s)

                # For first level, no need to choose parent
                if i == 0:
                    parent = self
                else:
                    # Find parent (closest node in tree)
                    parent = self._find_closest(node)

                parent.add_child(node)

        # If after all this, there are no `ObsNode` nodes,
        # then add a dummy.
        if len(self.get_obs_nodes()) == 0:
            self.add_child(DummyObsNode())

    @classmethod
    def synthetic(cls, stars, surveys):
        pass
