import lanelet2
import matplotlib.pyplot as plt
import numpy as np
import math
from lanelet2.core import BasicPoint2d, ConstLanelet, LaneletSequence
from lanelet2.geometry import length, to2D, toArcCoordinates
from lanelet2.projection import UtmProjector
from lanelet2.routing import LaneletPath
from scipy.interpolate import CubicSpline, UnivariateSpline


def is_iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True

class MapGeometry:
    def __init__(self, lanelet_filename=None, lat0=None, lon0=None):
        if lanelet_filename:
            self.projector = UtmProjector(lanelet2.io.Origin(lat0, lon0))
            self.lanelet_map = lanelet2.io.load(
                lanelet_filename, self.projector)
        else:
            self.getMapFromRos()

        self.traffic_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                                           lanelet2.traffic_rules.Participants.Vehicle)

        self.graph = lanelet2.routing.RoutingGraph(
            self.lanelet_map, self.traffic_rules)

        # Precompute lanelet interpolations
        self.interpolateCenterlines()

    def getNearestLanelets(self, p):
        matches = lanelet2.geometry.findNearest(
            self.lanelet_map.laneletLayer, p, 1)
        return matches[0]

    def getPossibleLaneletSequences(self, lanelet, maxDist):
        # Determine possible paths
        lanelet_paths = self.graph.possiblePaths(lanelet, maxDist)
        # Convert to lanelet sequence so geometric calculations are possible
        lanelet_sequences = [LaneletSequence(
            [lanelet for lanelet in lanelet_path]) for lanelet_path in lanelet_paths]

        return lanelet_sequences

    def projectToUTM(self, lat, lon, alt=0.0):
        gpsp = lanelet2.core.GPSPoint(lat, lon, alt)
        p3d = self.projector.forward(gpsp)
        return lanelet2.core.BasicPoint2d(p3d.x, p3d.y)

    def interpolateCenterlines(self):
        """Compute a continuously differentiable interpolation of lanelet centerlines for e.g. curvature calculation."""
        self.centerline_interpolations = {}

        for lanelet in self.lanelet_map.laneletLayer:
            (x_spl, y_spl) = self.interpolateLinestring(lanelet.centerline)

            # Save interpolations for later use
            self.centerline_interpolations[lanelet.id] = (x_spl, y_spl)

    def interpolateLinestring(self, linestring):
        # Get linestring 2d coordinates
        coords = np.array([(c.x, c.y) for c in linestring])
        # Calculate parametric length of each coordinate along linestring
        lengths = np.array([toArcCoordinates(
            to2D(linestring), BasicPoint2d(c[0], c[1])).length for c in coords])

        if len(lengths) > 3:
            # Univariate spline smoothly interpolates coordinates with putting high weight on start and end points
            def SplineFunction(x, y):
                weights = np.zeros((len(x)))
                # Set up weights so that first and last point are matched
                weights[0] = 1/3.0
                weights[-1] = 1/3.0
                weights[1:-1] = 1/3.0/(len(weights)-2)
                return UnivariateSpline(x, y, weights)
        else:
            # Cubic spline passes exactly through coordinates
            SplineFunction = CubicSpline

        # Interpolate one spline for x and y each
        x_spl = SplineFunction(lengths, coords[:, 0])
        y_spl = SplineFunction(lengths, coords[:, 1])

        # Return splines for this linestring
        return x_spl, y_spl

    def getMatchingLaneletInLaneletSequence(self, lanelet_sequence: LaneletSequence, match_length):
        """Determine matching lanelet in lanelet sequence based on given arc length.
        Multiple arc lengths have to be passed in ascending order."""
        if is_iterable(match_length):
            # Multiple match_lengths are tested
            cumulative_length = 0.0
            matched_lanelets = []
            last_matched_length_i = 0

            for lanelet in lanelet_sequence:
                cumulative_length += length(lanelet)

                for ml in match_length[last_matched_length_i:]:
                    if ml <= cumulative_length:
                        # If lanelet is longer than match length, we found a matching lanelet
                        matched_lanelets.append(lanelet)
                        last_matched_length_i += 1
                    else:
                        # If match length is longer, proceed to next lanelet
                        break

                if last_matched_length_i == len(match_length):
                    # If all match_lengths have been matched, stop searching
                    break

            if last_matched_length_i < len(match_length):
                # If not all match_lengths have been matched, assign last lanelet to them
                matched_lanelets += [lanelet_sequence[-1]] * \
                    (len(match_length) - last_matched_length_i)

            return matched_lanelets

        else:
            # Single match_length is tested
            cumulative_length = 0.0
            for lanelet in lanelet_sequence:
                cumulative_length += length(lanelet)

                # If lanelet is longer than match length, we found the matching lanelet
                if match_length <= cumulative_length:
                    return lanelet

            # If match length is larger than lanelet sequence length, return last lanelet
            return lanelet_sequence[-1]

    def calculateLaneletCurvature(self, lanelet_or_sequence, t):
        """Convenience function to calculate the curvature along a lanelet or lanelet sequence"""
        if type(lanelet_or_sequence) == ConstLanelet:
            lanelet = lanelet_or_sequence
            (x_spl, y_spl) = self.centerline_interpolations[lanelet.id]
            return self.calculateSplineCurvature(x_spl, y_spl, t)
        elif type(lanelet_or_sequence) == LaneletSequence:
            lanelet_sequence = lanelet_or_sequence

        else:
            raise NotImplementedError("Unsupported lanelet type")

    def calculateLaneletOrientation(self, lanelet_or_path, t):
        """Convenience function to calculate the orientation along a lanelet or lanelet sequence"""
        if type(lanelet_or_path) == ConstLanelet:
            (x_spl, y_spl) = self.centerline_interpolations[lanelet_or_path.id]
            return self.calculateSplineOrientation(x_spl, y_spl, t)
        elif type(lanelet_or_path) == LaneletPath:
            raise NotImplementedError("Unsupported lanelet type")
        else:
            print("Unsupported type")
            raise NotImplementedError("Unsupported lanelet type")

    def calculateSplineCurvature(self, x_spl, y_spl, t):
        """Returns signed curvature of spline in 2D plane for given parametric length t"""
        # Calculate spline derivatives
        x_p = x_spl(t, 1)
        x_pp = x_spl(t, 2)
        y_p = y_spl(t, 1)
        y_pp = y_spl(t, 2)

        # Formula from https://de.wikipedia.org/wiki/Kr%C3%BCmmung#Ebene_Kurven
        return np.divide(np.multiply(x_p, y_pp) - np.multiply(x_pp, y_p),
                         np.power(np.power(x_p, 2)+np.power(y_p, 2), 1.5))

    def calculateSplineOrientation(self, x_spl, y_spl, t):

        """Calculate signed orientation of spline in 2D plane for given parametric length t"""
        # Calculate spline derivatives
        x_p = x_spl(t, 1)
        y_p = y_spl(t, 1)

        return np.arctan2(y_p, x_p)

    def plotCenterlineCharacteristics(self, lanelet):
        # Get lanelet centerline 2d coordinates
        coords = np.array([(c.x, c.y) for c in lanelet.centerline])

        (x_spl, y_spl) = self.centerline_interpolations[lanelet.id]

        fig, ax = plt.subplots(3, 1, figsize=(6.5, 4))
        t = np.linspace(0, length(lanelet.centerline), 100)
        ax[0].set_title('Centerline and Interpolation')
        ax[0].plot(coords[:, 0], coords[:, 1], 'o-', label='centerline')
        ax[0].plot(x_spl(t), y_spl(t), label='spline')
        ax[0].axes.set_aspect('equal')
        ax[0].plot(coords[0, 0], coords[0, 1], 'ro')
        ax[0].legend()
        ax[1].set_xlabel('Length along centerline [m]')
        ax[1].set_ylabel('Curvature [rad/m]')
        ax[1].plot(t, self.calculateSplineCurvature(x_spl, y_spl, t))
        ax[2].set_xlabel('Length along centerline [m]')
        ax[2].set_ylabel('Orientation [rad]')
        ax[2].plot(t, self.calculateSplineOrientation(x_spl, y_spl, t))
        plt.show(block=True)

    # Lanelet map
    lanelet_map: lanelet2.core.LaneletMap = None

    # Spline interpolations to each lanelet's centerline
    centerline_interpolations: []