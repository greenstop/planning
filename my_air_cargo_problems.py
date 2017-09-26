from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph


class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions()

        self.state_index = { c:i for (i,c) in enumerate(self.state_map)};
        self.relaxed_actions_list = self.get_relaxed_actions();

    def get_relaxed_actions(self):
        '''
            Strip preconditions.
            Strip effects unless a goal fluent.
        '''
        from aimacode.utils import Expr
        relaxed = [];
        doAdd = False;
        for a in self.actions_list:
            effect_add = [ e for e in a.effect_add if e in self.goal];
            effect_rem = [ e for e in a.effect_rem if e in self.goal];
            if not effect_add and not effect_rem: continue;
            action = Action(
                Expr(a.name,*a.args),
                [[],[]],
                [effect_add,effect_rem]
                );
            relaxed.append(action);
        return relaxed;

    def get_actions(self):
        '''
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        '''

        def load_actions():
            '''Create all concrete Load actions and return a list

            :return: list of Action objects
            Example:
            precond_pos = [expr("Human(person)"), expr("Hungry(Person)")]
            precond_neg = [expr("Eaten(food)")]
            effect_add = [expr("Eaten(food)")]
            effect_rem = [expr("Hungry(person)")]
            eat = Action(expr("Eat(person, food)"), [precond_pos, precond_neg], [effect_add, effect_rem])
            '''
            loads = []
            for c in self.cargos:
                for p in self.planes:
                    for a in self.airports:
                        precond = [
                            expr("At(%s,%s)" % (c,a)),
                            expr("At(%s,%s)" % (p,a)),
                            ];
                        precondNot = [
                            ];
                        effect = [
                            expr("In(%s,%s)" % (c,p))
                            ];
                        effectNot = [
                            expr("At(%s,%s)" % (c,a))
                            ];
                        load = Action(
                            expr("Load(%s,%s,%s)" % (c,p,a)),
                            [precond,precondNot],
                            [effect,effectNot]
                            );
                        loads.append(load);
            return loads;

        def unload_actions():
            '''Create all concrete Unload actions and return a list

            :return: list of Action objects
            '''
            unloads = [];
            for c in self.cargos:
                for p in self.planes:
                    for a in self.airports:
                        precond = [
                            expr("In(%s,%s)" % (c,p)),
                            expr("At(%s,%s)" % (p,a)),
                            ];
                        precondNot = [
                            ];
                        effect = [
                            expr("At(%s,%s)" % (c,a))
                            ];
                        effectNot = [
                            expr("In(%s,%s)" % (c,p))
                            ];
                        unload = Action(
                            expr("Unload(%s,%s,%s)" % (c,p,a)),
                            [precond,precondNot],
                            [effect,effectNot]
                            );
                        unloads.append(unload);
            return unloads;

        def fly_actions():
            '''Create all concrete Fly actions and return a list

            :return: list of Action objects
            '''
            flys = []
            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for p in self.planes:
                            precond_pos = [expr("At({}, {})".format(p, fr)),
                                           ]
                            precond_neg = []
                            effect_add = [expr("At({}, {})".format(p, to))]
                            effect_rem = [expr("At({}, {})".format(p, fr))]
                            fly = Action(expr("Fly({}, {}, {})".format(p, fr, to)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        return load_actions() + unload_actions() + fly_actions()

    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """

        def check_precondition(action):
            for fluent in action.precond_pos:
                index = self.state_index[fluent];
                if state[index] == 'F':
                    return False;
            for fluent in action.precond_neg:
                index = self.state_index[fluent];
                if state[index] == 'T':
                    return False;
            return True;

        possible_actions = []
        for action in self.actions_list:
            if check_precondition(action) is True:
                possible_actions.append(action);

        return possible_actions

    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        #make false the effect_rem
        for fluent in action.effect_rem:
            index = self.state_map.index(fluent);
            state = state[:index]+"F"+state[index+1:];
        #make true the effect_add
        for fluent in action.effect_add:
            index = self.state_map.index(fluent);
            state = state[:index]+"T"+state[index+1:];
        return state;

    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    def h_pg_levelsum(self, node: Node):
        '''
        This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        '''
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    def h_ignore_preconditions(self, node: Node):
        '''
        This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        '''
        count = 0
        relaxed = AirCargoProblem(self.cargos,
                                  self.planes,
                                  self.airports,
                                  FluentState([],[]),
                                  self.goal)
        relaxed.initial = node.state;
        relaxed.initial_state_TF = node.state;
        relaxed.state_map = self.state_map;
        relaxed.actions_list = self.relaxed_actions_list;

        from aimacode.search import (Node,breadth_first_search);
        result = breadth_first_search(relaxed);
        return len(result.solution()) if result else float("inf");


def air_cargo_p1() -> AirCargoProblem:
    cargos = ['C1', 'C2']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = [expr('At(C2, SFO)'),
           expr('In(C2, P1)'),
           expr('In(C2, P2)'),
           expr('At(C1, JFK)'),
           expr('In(C1, P1)'),
           expr('In(C1, P2)'),
           expr('At(P1, JFK)'),
           expr('At(P2, SFO)'),
           ]
    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p2() -> AirCargoProblem:
    cargos = ['C1', 'C2', 'C3']
    planes = ['P1', 'P2', 'P3']
    airports = ['JFK', 'SFO', 'ATL']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           expr('At(P3, ATL)'),
           ]
    neg = []  + [ expr('At(%s,%s)'%(c,a)) for c in cargos for a in airports ] 
    neg = neg + [ expr('At(%s,%s)'%(p,a)) for p in planes for a in airports ] 
    neg = neg + [ expr('In(%s,%s)'%(c,p)) for c in cargos for p in planes ] 

    for clause in pos:
        neg.pop(neg.index(clause));

    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p3() -> AirCargoProblem:
    cargos = ['C1', 'C2', 'C3', 'C4']
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO', 'ATL', 'ORD']
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(C4, ORD)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]
    neg = []  + [ expr('At(%s,%s)'%(c,a)) for c in cargos for a in airports ] 
    neg = neg + [ expr('At(%s,%s)'%(p,a)) for p in planes for a in airports ] 
    neg = neg + [ expr('In(%s,%s)'%(c,p)) for c in cargos for p in planes ] 

    for clause in pos:
        neg.pop(neg.index(clause));

    init = FluentState(pos, neg)
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, JFK)'),
            expr('At(C4, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)
