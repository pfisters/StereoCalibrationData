from absl import app, logging, flags
from absl.flags import FLAGS, argparse_flags
from src.FactorySettings import FactorySettings
from src.Scene import Scene, SceneType
from src.Scenario import Scenario
from src.Change import Change, ChangeType

FLAGS = flags.FLAGS
flags.DEFINE_float('X', -500, 'cube: Tx')
flags.DEFINE_float('Y', -500, 'cube: Ty')
flags.DEFINE_float('Z', 3000, 'cube: Tz')
flags.DEFINE_float('SCALE', 500, 'cube: -')
flags.DEFINE_float('DISTANCE', 3000, 'cube: -')
flags.DEFINE_float('ROT_X', 5, 'cube: Rx')
flags.DEFINE_float('ROT_Y', -4, 'cube: Ry')
flags.DEFINE_float('ROT_Z', 10, 'cube: Rz')
flags.DEFINE_float('MIN', 0, 'cube: point lower boundary')
flags.DEFINE_float('MAX', 2000, 'cube: point upper boundary')
flags.DEFINE_integer('POINTS', 500, 'Number of points in scene')


def main(argv):

    # read factory settings
    factory_settings = FactorySettings('factory_settings.json')

    # create synthetic scene
    scene = {
        'X' : FLAGS.X,
        'Y' : FLAGS.Y,
        'Z' : FLAGS.Z,
        'SCALE' : FLAGS.SCALE,
        'DISTANCE' : FLAGS.DISTANCE,
        'ROT_X' : FLAGS.ROT_X,
        'ROT_Y' : FLAGS.ROT_Y,
        'ROT_Z' : FLAGS.ROT_Z,
        'MIN' : FLAGS.MIN,
        'MAX' : FLAGS.MAX,
        'POINTS': FLAGS.POINTS
    }
    points = Scene(SceneType.cube, scene)
    points.visualize()

    logging.info('Create Scencario')

    changes = []
    changes += [Change(ChangeType.translation, 20, [10, 0, 0])]
    changes += [Change(ChangeType.translation, 30, [0, 10, 0])]
    changes += [Change(ChangeType.translation, 40, [0 ,0, 10])]
    changes += [Change(ChangeType.rotation, 50, [3, 0, 0])]
    changes += [Change(ChangeType.rotation, 60, [0, 3, 0])]
    changes += [Change(ChangeType.rotation, 70, [0, 0, 3])]

    # create scneario
    scenario = Scenario(100, changes, points, factory_settings, (752, 480), 2.)
    scenario.generate_sequence()


if __name__ == "__main__":
    try:
        app.run(main)
    except SystemExit:
        pass