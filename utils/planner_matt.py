from ompl import util as ou
from ompl import base as ob
from ompl import geometric as og
import math
import sys
import numpy as np
import pybullet as p
import os
import time


class ValidityChecker(ob.StateValidityChecker):
    def __init__(self, si, obstacle_ids):
        super(ValidityChecker, self).__init__(si)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        current_dir = current_dir.replace("utils", "env/models")
        self.gripper_id = p.loadURDF(os.path.join(current_dir, "tm5_900/robotiq_85.urdf"),
                                     [-5., 0., 0.1],
                                     useFixedBase=True)
        self.obstacle_ids = obstacle_ids
    # Returns whether the given state's position overlaps the
    # circular obstacle
    def isValid(self, state):
        return self.clearance(state)

    # Returns the distance from the given state's position to the
    # boundary of the circular obstacle.
    def clearance(self, state):
        # Extract the robot's (x,y) position from its state
        x = state.getX()
        y = state.getY()
        z = state.getZ()
        qx = state.rotation().x
        qy = state.rotation().y
        qz = state.rotation().z
        qw = state.rotation().w
        p.resetBasePositionAndOrientation(self.gripper_id, [x, y, z],
                                          [qx, qy, qz, qw])
        
        for obstacle_id in self.obstacle_ids:
            contact_points = p.getClosestPoints(self.gripper_id, obstacle_id, distance=0.001)
            if len(contact_points) > 0:
                return False
        # time.sleep(0.05)
        return True


def getPathLengthObjective(si):
    return ob.PathLengthOptimizationObjective(si)


def getThresholdPathLengthObj(si):
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostThreshold(ob.Cost(1.51))
    return obj


class ClearanceObjective(ob.StateCostIntegralObjective):
    def __init__(self, si):
        super(ClearanceObjective, self).__init__(si, True)
        self.si_ = si

    # Our requirement is to maximize path clearance from obstacles,
    # but we want to represent the objective as a path cost
    # minimization. Therefore, we set each state's cost to be the
    # reciprocal of its clearance, so that as state clearance
    # increases, the state cost decreases.
    def stateCost(self, s):
        return ob.Cost(1 / (self.si_.getStateValidityChecker().clearance(s) +
                       sys.float_info.min))


def getClearanceObjective(si):
    return ClearanceObjective(si)


def getBalancedObjective1(si):
    lengthObj = ob.PathLengthOptimizationObjective(si)
    clearObj = ClearanceObjective(si)

    opt = ob.MultiOptimizationObjective(si)
    opt.addObjective(lengthObj, 5.0)
    opt.addObjective(clearObj, 1.0)

    return opt


def getPathLengthObjWithCostToGo(si):
    obj = ob.PathLengthOptimizationObjective(si)
    obj.setCostToGoHeuristic(ob.CostToGoHeuristic(ob.goalRegionCostToGo))

    return obj


# Keep these in alphabetical order and all lower case
def allocatePlanner(si, plannerType):
    if plannerType.lower() == "bfmtstar":
        return og.BFMT(si)
    elif plannerType.lower() == "bitstar":
        return og.BITstar(si)
    elif plannerType.lower() == "fmtstar":
        return og.FMT(si)
    elif plannerType.lower() == "informedrrtstar":
        return og.InformedRRTstar(si)
    elif plannerType.lower() == "prmstar":
        return og.PRMstar(si)
    elif plannerType.lower() == "rrtstar":
        return og.RRTstar(si)
    elif plannerType.lower() == "sorrtstar":
        return og.SORRTstar(si)
    elif plannerType.lower() == "rrtconnect":
        return og.RRTConnect(si)
    else:
        ou.OMPL_ERROR("Planner-type is not implemented in allocation function.")


# Keep these in alphabetical order and all lower case
def allocateObjective(si, objectiveType):
    if objectiveType.lower() == "pathclearance":
        return getClearanceObjective(si)
    elif objectiveType.lower() == "pathlength":
        return getPathLengthObjective(si)
    elif objectiveType.lower() == "thresholdpathlength":
        return getThresholdPathLengthObj(si)
    elif objectiveType.lower() == "weightedlengthandclearancecombo":
        return getBalancedObjective1(si)
    else:
        ou.OMPL_ERROR("Optimization-objective is not implemented in allocation function.")


class GraspPlanner():
    def __init__(self,
                 planner_type="RRTstar",
                 objectiveType="PathLength",
                 upper_bound=[1, 0.8, 1.4],
                 lower_bound=[-0., -0.8, 0],
                 obstacles = None
                 ) -> None:
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.space = ob.SE3StateSpace()
        self.set_space_bound()
        self.si = ob.SpaceInformation(self.space)
        self.validityChecker = ValidityChecker(self.si, obstacles)
        self.optimizingPlanner = allocatePlanner(self.si, planner_type)
        self.si.setStateValidityChecker(self.validityChecker)
        self.si.setup()
        self.optim_obj = allocateObjective(self.si, objectiveType)

    def set_space_bound(self):
        bound = ob.RealVectorBounds(3)
        bound.setHigh(0, self.upper_bound[0])
        bound.setHigh(1, self.upper_bound[1])
        bound.setHigh(2, self.upper_bound[2])

        bound.setLow(0, self.lower_bound[0])
        bound.setLow(1, self.lower_bound[1])
        bound.setLow(2, self.lower_bound[2])
        self.space.setBounds(bound)

    def plan(self, init_pos, goal_pos, path_length=20, runTime=1.0, planner_range=0.):
        start = ob.State(self.space)
        goal = ob.State(self.space)
        for i in range(7):
            start[i] = init_pos[i]
            goal[i] = goal_pos[i]

        start.enforceBounds()
        goal.enforceBounds()

        if not start.satisfiesBounds() or not goal.satisfiesBounds():
            print("Start of Goal position is invalid")
            return None

        # Create a problem instance
        pdef = ob.ProblemDefinition(self.si)

        # Set the start and goal states
        pdef.setStartAndGoalStates(start, goal)

        # Create the optimization objective specified by our command-line argument.
        # This helper function is simply a switch statement.
        pdef.setOptimizationObjective(self.optim_obj)

        # Construct the optimal planner specified by our command line argument.
        # This helper function is simply a switch statement.

        # Set the problem instance for our planner to solve
        self.optimizingPlanner.clear()
        self.optimizingPlanner.setProblemDefinition(pdef)
        self.optimizingPlanner.setRange(planner_range)
        self.optimizingPlanner.setup()

        # attempt to solve the planning problem in the given runtime
        solved = self.optimizingPlanner.solve(runTime)
        if solved:
            # Output the length of the path found
            print('{0} found solution of path length {1:.4f} with an optimization objective value of {2:.4f}'.format(
                self.optimizingPlanner.getName(),
                pdef.getSolutionPath().length(),
                pdef.getSolutionPath().cost(pdef.getOptimizationObjective()).value()))

            if pdef.getSolutionPath().getStateCount() < path_length:
                pdef.getSolutionPath().interpolate(path_length)
                print(f"Interpolate Path length to {path_length}")
            p.resetBasePositionAndOrientation(self.validityChecker.gripper_id, [-5., 0., 0.1], [0, 0, 0, 1])
            return pdef
        else:
            print("No solution found.")
            p.resetBasePositionAndOrientation(self.validityChecker.gripper_id, [-5., 0., 0.1], [0, 0, 0, 1])
            return None
    
    def show_path(self, gripper_pos_path, gripper_ori_path):
        for pos, ori in zip(gripper_pos_path, gripper_ori_path):
            p.resetBasePositionAndOrientation(self.validityChecker.gripper_id, pos, ori)
            time.sleep(0.1)
        p.resetBasePositionAndOrientation(self.validityChecker.gripper_id, [-5., 0., 0.1], [0, 0, 0, 1])