from docopt import docopt
from local_library.ExtractLandmarks import ExtractLandmarks
from local_library.ProcessVideo import ProcessVideo
from local_library.MachineLearning import MachineLearning

command_help = """
Voici quelques informations sur comment utiliser cette application

Usage:
    app.py extract video [shapefaciallandmarks] [-g | -n] <file> <save_repertory> [-v]
    app.py extract image_collection [shapefaciallandmarks] [-g | -n] <path> <save_repertory> [-v]
    app.py learn (gmm | svm) <file_normal> [<file_grimace>] [<file_predict>] [-v]
    app.py predict (-g | -n | -t) <file_clf> <file_content> [-v]

Options:
    -h --help       Affiche cette aide
    -g --grimaces   Indique que la source (video ou collection d'images) est basée sur des grimaces
    -n --normaux    Indique que la source (video ou collection d'images) est basée sur des expressions neutres
    -t --tests      Permet de lancer des tests ?
    -v --verbose    Permet de lancer l'application en mode verbeux
"""


if __name__ == "__main__":
    # Arguments
    arguments = docopt(command_help)

    # Main
    if arguments['extract']:
        video = None
        process_video = ProcessVideo(extract_shape_facial_landmarks=arguments['shapefaciallandmarks'])

        if arguments['image_collection']:
            video = ExtractLandmarks(source_type='image_collection', file=arguments["<path>"],
                                     save_repertory=arguments["<save_repertory>"], verbose=arguments['--verbose'])
        elif arguments['video']:
            video = ExtractLandmarks(source_type='video', file=arguments["<file>"],
                                     save_repertory=arguments["<save_repertory>"], verbose=arguments['--verbose'])
        else:
            assert "[ERREUR] Impossible d'extraire si aucune source n'est fournie... (image_collection | video)"

        video.read_file(process_video.process_one_face_only)

        if arguments['--grimaces']:
            video.save_stream('facial_landmarks', 0)
        elif arguments['--normaux']:
            video.save_stream('facial_landmarks', 1)
        else:
            assert "[ERREUR] Merci de préciser si la source présente des grimaces ou des réactions neutres..."

    elif arguments['learn']:
        if arguments['<file_normal>']:
            if arguments['<file_grimace>']:
                ml = MachineLearning(arguments["<file_normal>"],
                                     arguments["<file_grimace>"])
            else:
                ml = MachineLearning(arguments["<file_normal"])

        if arguments["svm"]:
            ml.svm_classifier()
        else:
            ml.gmm_classifier()

        ml.save_clf()
        if arguments["<file_predict>"]:
            print(["! Prediction :"])
            ml.predict(arguments["<file_predict>"])

    elif arguments['predict']:
        assert "[ERREUR] Cette fonction n'est pas encore mise en place"

    else:
        assert "[ERREUR] Aucune fonctionnalité choisie... (extract | learn | predict)"
