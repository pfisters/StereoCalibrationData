from absl import app, logging
from FactorySettings import FactorySettings
from Scene import Scene, SceneType
from Scenario import Scenario
from Change import Change, ChangeType

def main(argv):

    # read factory settings
    factory_settings = FactorySettings('factory_settings.json')

    # create synthetic scene
    logging.info('Create Scene')
    scene = {
        'X' : -500,
        'Y' : -500,
        'Z' : 3000,
        'SCALE' : 500,
        'DISTANCE' : 3000,
        'ROT_X' : 5,
        'ROT_Y' : -4,
        'ROT_Z' : 10,
        'MIN' : 0,
        'MAX' : 2000,
        'POINTS': 500
    }
    pts = Scene(SceneType.cube, scene)
    pts.visualize()

    logging.info('Create Scencario')

    changes = []
    changes += [Change(ChangeType.translation, 20, [10, 0, 0])]
    changes += [Change(ChangeType.translation, 30, [0, 10, 0])]
    changes += [Change(ChangeType.translation, 40, [0 ,0, 10])]
    changes += [Change(ChangeType.rotation, 50, [3, 0, 0])]
    changes += [Change(ChangeType.rotation, 60, [0, 3, 0])]
    changes += [Change(ChangeType.rotation, 70, [0, 0, 3])]

    # create scneario
    scenario = Scenario(100, changes, pts, factory_settings, (752, 480), 2.)
    scenario.generate_sequence()


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass
