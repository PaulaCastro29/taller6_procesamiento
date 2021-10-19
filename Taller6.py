#-----------------------------------------
#TALLER 6: PAULA CASTRO Y MICHAEL CONTRERAS
#-----------------------------------------
import cv2
import numpy as np
import os
import imutils

#Definición de método puntos de interes, aqui se debe definir si desea que sea por ORB o por SIFT
feature_extractor = 'orb' # 'sift'
feature_matching = 'bf'

if __name__ == '__main__':

    # Se cargan todas las imagenes contenidas en el folder
    input_images_path = "C:/CodigoPyCharmPaula/proyecciones/imagenes"
    files_names = os.listdir(input_images_path)

    # Se le muestra al usuario la cantidad de imagenes y su orden
    print("El número de imagenes es: ", len(files_names))
    print("las imagenes son:", files_names)

    # Se le solicita al usuario seleccionar la imagen de referencia para aplicar las homografias y se valida que este dentro de la cantidad de imagenes
    N = int(input('Ingrese el número de la imagen que quiere tomar como referencia:'))
    if N >= 1 and N <= len(files_names):
        N = N
        # print(N)
    else:
        print("Error! el número ingresado supera el número de imagenes")
    # De acuerdo con la imagen de referencia seleccionada se aplica la homografia al primer par de imagenes Nota. en este código solo se ve el caso en el que la imagen de referencia es 2.
    if N == 2:
        lista = []
        for file_name in files_names:
            image_path = input_images_path + "/" + file_name
            lista.append(image_path)

        image_name_1 = lista[1]
        image_name_2 = lista[0]
        image_name_3 = lista[2]
        trainImg = cv2.imread(image_name_1)
        trainImg = cv2.resize(trainImg, (700, 700), interpolation=cv2.INTER_CUBIC)
        trainImg_gray = cv2.cvtColor(trainImg, cv2.COLOR_BGR2GRAY)
        queryImg = cv2.imread(image_name_2)
        queryImg = cv2.resize(queryImg, (700, 700), interpolation=cv2.INTER_CUBIC)
        queryImg_gray = cv2.cvtColor(queryImg, cv2.COLOR_BGR2GRAY)
        queryImg2 = cv2.imread(image_name_3)
        queryImg2 = cv2.resize(queryImg2, (700, 700), interpolation=cv2.INTER_CUBIC)
        queryImg_gray2 = cv2.cvtColor(queryImg2, cv2.COLOR_BGR2GRAY)

        #Método que detecta los puntos de interes y define los descriptores
        def detectAndDescribe(image, method):
            if method == 'sift':
                descriptor = cv2.SIFT_create()
            elif method == 'orb':
                descriptor = cv2.ORB_create()
            (kps, features) = descriptor.detectAndCompute(image, None)
            return (kps, features)

        kpsA, featuresA = detectAndDescribe(trainImg_gray, method=feature_extractor)
        kpsB, featuresB = detectAndDescribe(queryImg_gray, method=feature_extractor)
        kpsC, featuresC = detectAndDescribe(queryImg_gray2, method=feature_extractor)

        #Método que define los mejores puntos entre las dos imagenes comparadas
        def createMatcher(method, crossCheck):
            if method == 'sift':
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=crossCheck)
            elif method == 'orb':
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=crossCheck)
            return bf

        #Ordena por distancia los puntos los de menor distancia son más similares
        def matchKeyPointsBF(featuresA, featuresB, method):
            bf = createMatcher(method, crossCheck=True)
            best_matches = bf.match(featuresA, featuresB)
            rawMatches = sorted(best_matches, key=lambda x: x.distance)
            return rawMatches

        #Computa los puntos y los pone en una lista
        def matchKeyPointsKNN(featuresA, featuresB, ratio, method):
            bf = createMatcher(method, crossCheck=False)
            rawMatches = bf.knnMatch(featuresA, featuresB, 2)

            matches = []

            for m, n in rawMatches:
                if m.distance < n.distance * ratio:
                    matches.append(m)
            return matches

        if feature_matching == 'bf':
            matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor)
            img3 = cv2.drawMatches(trainImg, kpsA, queryImg, kpsB, matches[:100],
                                   None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        elif feature_matching == 'knn':
            matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor)
            img3 = cv2.drawMatches(trainImg, kpsA, queryImg, kpsB, np.random.choice(matches, 100),
                                   None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imshow("img3", img3)
        cv2.waitKey(0)

        #Se realiza la homografia entre la imagen 1 y 2
        def getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
            # convert the keypoints to numpy arrays
            kpsA = np.float32([kp.pt for kp in kpsA])
            kpsB = np.float32([kp.pt for kp in kpsB])

            if len(matches) > 4:

                ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
                ptsB = np.float32([kpsB[m.trainIdx] for m in matches])

                (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                                 reprojThresh)

                return (matches, H, status)
            else:
                return None

        M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=4)
        if M is None:
            print("Error!")
        (matches, H, status) = M
        #print(H)
        width = trainImg.shape[1] + queryImg.shape[1]
        height = trainImg.shape[0] + queryImg.shape[0]

        result = cv2.warpPerspective(trainImg, H, (width, height))
        result[0:queryImg.shape[0], 0:queryImg.shape[1]] = queryImg
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        #ELIMINA BORDES NEGROS
        #Encuentra los contornos de una imagen binaria
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        #Toma el maximo contorno de area
        c = max(cnts, key=cv2.contourArea)

        (x, y, w, h) = cv2.boundingRect(c)

        #Recorta la imagen
        result = result[y:y + h, x:x + w]

        resultRR = cv2.resize(result, (700, 700), interpolation=cv2.INTER_CUBIC)
        cv2.imshow("result", result)
        cv2.waitKey(0)

    # Homografia 2: Se realiza el mismo proceso anterior pero para la imagen 2 y 3

        kpsA, featuresA = detectAndDescribe(queryImg_gray2, method=feature_extractor)
        kpsB, featuresB = detectAndDescribe(resultRR, method=feature_extractor)

        def matchKeyPointsBF2(featuresA, featuresB, method):
            bf = createMatcher(method, crossCheck=True)
            best_matches = bf.match(featuresA, featuresB)
            rawMatches = sorted(best_matches, key=lambda x: x.distance)
            return rawMatches


        def matchKeyPointsKNN2(featuresA, featuresB, ratio, method):
            bf = createMatcher(method, crossCheck=False)
            rawMatches = bf.knnMatch(featuresA, featuresB, 2)

            matches = []

            for m, n in rawMatches:
                if m.distance < n.distance * ratio:
                    matches.append(m)
            return matches


        if feature_matching == 'bf':
            matches = matchKeyPointsBF(featuresA, featuresB, method=feature_extractor)
            img4 = cv2.drawMatches(queryImg2, kpsA, resultRR, kpsB, matches[:100],
                                   None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        elif feature_matching == 'knn':
            matches = matchKeyPointsKNN(featuresA, featuresB, ratio=0.75, method=feature_extractor)
            img4 = cv2.drawMatches(queryImg2, kpsA, resultRR, kpsB, np.random.choice(matches, 100),
                                   None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        cv2.imshow("img4", img4)
        cv2.waitKey(0)

        def getHomography2(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh):
            kpsA = np.float32([kp.pt for kp in kpsA])
            kpsB = np.float32([kp.pt for kp in kpsB])

            if len(matches) > 4:
                ptsA = np.float32([kpsA[m.queryIdx] for m in matches])
                ptsB = np.float32([kpsB[m.trainIdx] for m in matches])
                (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
                                                 reprojThresh)
                return (matches, H, status)
            else:
                return None

        M = getHomography(kpsA, kpsB, featuresA, featuresB, matches, reprojThresh=4)
        if M is None:
            print("Error!")
        (matches, H, status) = M
        width = queryImg2.shape[1] + resultRR.shape[1]
        height = queryImg2.shape[0] + resultRR.shape[0]

        result2 = cv2.warpPerspective(queryImg2, H, (width, height))
        result2[0:resultRR.shape[0], 0:resultRR.shape[1]] = resultRR

        gray = cv2.cvtColor(result2, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        c = max(cnts, key=cv2.contourArea)

        (x, y, w, h) = cv2.boundingRect(c)

        result2 = result2[y:y + h, x:x + w]

# Resultado final del stitching
        cv2.imshow("result2", result2)
        cv2.imwrite('Final.png', result2)
        cv2.waitKey(0)

    else:
       print ("Error! el número ingresado es diferente de 2 para efectos del taller utilice el 2 para ejecutar el caso")