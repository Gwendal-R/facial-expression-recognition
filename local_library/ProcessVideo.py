import pandas as pd


class ProcessVideo:
    def __init__(self, extract_shape_facial_landmarks=False):
        self.extract_shape_facial_landmarks = extract_shape_facial_landmarks
        self.FACEPARTSNAME = ["jaw", "left_eyebrow", "nose", "mouth", "right_eyebrow", "left_eye", "right_eye"]

    def process_one_face_only(self, video_player, frame):
        # Extract face
        frame.extract_shape_faces()

        # Extract landmarks into image
        if len(frame.faces) == 1:
            nface = 0

            if self.extract_shape_facial_landmarks:
                video_player.generate_image("frame" + str(frame.nframe) + ".face" + str(nface),
                                            frame.faces[nface].generate_shape())

            # Extract landmarks into csv
            landmarks = []
            for landmarks_name in self.FACEPARTSNAME:
                facial_landmark = frame.faces[nface].extract_facial_landmarks(landmarks_name)
                landmarks.append(pd.Series(facial_landmark['value'], index=facial_landmark['column']))

            landmarks.append(pd.Series([frame.nframe], index=['nframe']))
            video_player.add_stream('facial_landmarks', pd.concat(landmarks))

        elif len(frame.faces) > 1:
            assert "[ERREUR] Ce logiciel ne supporte, pour l'instant que les vidéos contenant un seul visage..."

        else:
            assert "[ERREUR] Aucun visage détectable sur la vidéo..."

        frame.release()
