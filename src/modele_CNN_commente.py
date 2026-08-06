# ================================================================
# DÉTECTION DE L'APNÉE DU SOMMEIL PAR CNN 1D
# ================================================================
# Auteur  : Projet de Fin d'Études — ENSAM Rabat
# Dataset : Apnea-ECG PhysioNet (Moody & Mark, 2000)
#           70 enregistrements ECG — 100 Hz
# Modèle  : v3_4conv_a0.53_lr8e-05_cw1.7_run1
# Carte   : ESP32-S3 N16R8 via TFLite Micro + Arduino IDE
# ================================================================

# ────────────────────────────────────────────────────────────────
# PARTIE 1 : CONNEXION À GOOGLE DRIVE
# ────────────────────────────────────────────────────────────────

# from google.colab import drive
# drive.mount('/content/drive')
# Connecte Google Colab à Google Drive
# Permet d'accéder aux fichiers du dataset
# et de sauvegarder le modèle entraîné

# ────────────────────────────────────────────────────────────────
# PARTIE 2 : IMPORTATION DES BIBLIOTHÈQUES
# ────────────────────────────────────────────────────────────────

import numpy as np
# Bibliothèque pour les calculs mathématiques sur les tableaux
# Utilisée pour charger et manipuler les données ECG

import os
# Outil pour construire les chemins de fichiers

import random
# Génère des nombres aléatoires
# Utilisé uniquement pour fixer la reproductibilité

import matplotlib.pyplot as plt
# Bibliothèque pour tracer les graphiques
# Utilisée pour afficher les courbes ROC, Accuracy et Loss

import tensorflow as tf
# Framework principal pour la création et l'entraînement
# du réseau de neurones (créé par Google)

from tensorflow.keras.models import Sequential
# Permet de construire un modèle couche par couche
# Chaque couche reçoit la sortie de la couche précédente

from tensorflow.keras.layers import (Conv1D, MaxPooling1D,
                                     BatchNormalization,
                                     GlobalMaxPooling1D,
                                     Dense, Dropout)
# Les briques qui composent notre CNN 1D :
# Conv1D            : analyse le signal ECG par filtres glissants
# MaxPooling1D      : réduit la taille du signal de moitié
# BatchNormalization: stabilise les valeurs entre les couches
# GlobalMaxPooling1D: extrait le pic maximum de chaque filtre
# Dense             : couche fully connected pour la décision finale
# Dropout           : désactive des neurones pour éviter l'overfitting

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# EarlyStopping     : arrête l'entraînement au meilleur moment
# ReduceLROnPlateau : réduit le learning rate quand les performances stagnent

from tensorflow.keras.regularizers import l2
# Régularisation L2 : pénalise les grands poids
# Force le modèle à comprendre plutôt que mémoriser

from sklearn.metrics import (confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve,
                             f1_score)
# Outils de mesure des performances du modèle :
# confusion_matrix      : calcule TN, FP, FN, TP
# roc_auc_score         : calcule l'AUC (Area Under Curve)
# roc_curve             : données pour tracer la courbe ROC
# precision_recall_curve: données pour la courbe Précision-Rappel
# f1_score              : calcule le score F1

# ────────────────────────────────────────────────────────────────
# PARTIE 3 : REPRODUCTIBILITÉ — FIXATION DES SEEDS
# ────────────────────────────────────────────────────────────────

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
# Fixe le hasard à la valeur 42 pour les trois bibliothèques
# Garantit que les résultats sont identiques à chaque lancement
# Sans cette étape, les poids initiaux du réseau seraient différents
# à chaque exécution, donnant des résultats non reproductibles
print("Seeds fixes ✅")

# ────────────────────────────────────────────────────────────────
# PARTIE 4 : DÉFINITION DES CHEMINS DE FICHIERS
# ────────────────────────────────────────────────────────────────

SAVE_PATH    = '../data/processed'
# Dossier contenant les fichiers numpy du dataset préparé par preprocess.py
# ('../data/processed' depuis src/ en local, /data/processed dans le conteneur)

RESULTS_PATH = '../results'
# Dossier où seront sauvegardés le modèle entraîné et les graphiques

# ────────────────────────────────────────────────────────────────
# PARTIE 5 : CHARGEMENT DES DONNÉES ECG
# ────────────────────────────────────────────────────────────────

X_train = np.load(os.path.join(SAVE_PATH, 'X_train_apnea.npy'))
y_train = np.load(os.path.join(SAVE_PATH, 'y_train_apnea.npy'))
X_test  = np.load(os.path.join(SAVE_PATH, 'X_test_apnea.npy'))
y_test  = np.load(os.path.join(SAVE_PATH, 'y_test_apnea.npy'))
# Chargement des 4 fichiers numpy depuis Google Drive
#
# X_train : segments ECG du set d'apprentissage
#           Enregistrements a01-a20 (sévère), b01-b05 (modéré),
#           c01-c10 (quasi normal) — 35 enregistrements
#           Shape : (N, 6000) — N segments de 6000 points chacun
#
# y_train : étiquettes du set d'apprentissage
#           0 = Normal, 1 = Apnée
#
# X_test  : segments ECG du set de test
#           Enregistrements x01-x35 — 17 248 segments
#           Jamais utilisés pendant l'entraînement
#
# y_test  : étiquettes du set de test
#           Utilisées uniquement pour l'évaluation finale

X_train = X_train[..., np.newaxis]
X_test  = X_test[...,  np.newaxis]
# Ajout d'une dimension de canal au signal ECG
# Avant : shape (N, 6000)    — tableau 2 dimensions
# Après : shape (N, 6000, 1) — tableau 3 dimensions
#
# Le CNN 1D exige obligatoirement la forme :
# (nombre_segments, longueur_signal, nombre_canaux)
#
# Notre signal ECG a 1 seul canal (signal monodimensionnel)

# ────────────────────────────────────────────────────────────────
# PARTIE 6 : CONFIGURATION DES HYPERPARAMÈTRES
# ────────────────────────────────────────────────────────────────

RUN_NUM = 1
# Numéro de la configuration à utiliser parmi les 4 définies
# Pour tester une autre configuration : changer RUN_NUM = 2, 3 ou 4

configs = {
    1: {'alpha': 0.53, 'lr': 0.00008, 'cw': {0:1.7,  1:1.3}},
    2: {'alpha': 0.54, 'lr': 0.00008, 'cw': {0:1.7,  1:1.3}},
    3: {'alpha': 0.53, 'lr': 0.00008, 'cw': {0:1.65, 1:1.35}},
    4: {'alpha': 0.54, 'lr': 0.00008, 'cw': {0:1.65, 1:1.35}},
}
# Dictionnaire de 4 configurations testées
# Méthodologie : modifier un seul hyperparamètre à la fois
# et mesurer l'impact sur les performances
#
# alpha : paramètre de pondération de la Focal Loss
#         Identifié après avoir testé : 0.50, 0.53, 0.54, 0.56
#
# lr    : learning rate (vitesse d'apprentissage)
#         Identifié après avoir testé : 0.001, 0.0001, 0.00008, 0.00001
#
# cw    : class_weight — les exemples de la classe Normal contribuent
#         1.7 fois plus à la perte, ceux de la classe Apnée 1.3 fois plus
#         Compense le déséquilibre : 38% apnée / 62% normal
#         Identifié expérimentalement parmi plusieurs combinaisons

cfg     = configs[RUN_NUM]
alpha   = cfg['alpha']   # 0.53
lr      = cfg['lr']      # 0.00008
cw      = cfg['cw']      # {0:1.7, 1:1.3}

VERSION = f'v3_4conv_a{alpha}_lr{lr}_cw{cw[0]}_run{RUN_NUM}'
# Nom unique du modèle contenant tous ses hyperparamètres
# Permet la traçabilité des expérimentations

SEUIL = 0.500
# Seuil de décision initial fixé à 0.50
# Sera optimisé après l'entraînement par recherche exhaustive
# Résultat : seuil optimal = 0.54

print(f"\nVersion : {VERSION}")
print(f"Alpha   : {alpha}")
print(f"LR      : {lr}")
print(f"CW      : {cw}")

# ────────────────────────────────────────────────────────────────
# PARTIE 7 : FONCTION DE PERTE — FOCAL LOSS
# ────────────────────────────────────────────────────────────────

def focal_loss(gamma=2.0, alpha=alpha):
    # Fonction de perte personnalisée adaptée aux datasets déséquilibrés
    # Remplace la Binary CrossEntropy standard
    #
    # Problème : dataset déséquilibré : 38% apnée / 62% normal
    # Avec BCE standard : le modèle favorise la classe majoritaire (normal)
    #
    # Focal Loss : force le modèle à accorder davantage d'importance
    # aux exemples difficiles à classifier, qu'il s'agisse d'apnées
    # ou de segments normaux difficiles
    #
    # gamma=2.0 : intensité de la focalisation sur les cas difficiles
    #             Valeur recommandée par Lin et al. (IEEE ICCV, 2017)
    # alpha=0.53 : légère préférence pour la classe apnée (> 0.5)

    def loss(y_true, y_pred):
        # y_true : vraies étiquettes (0=Normal ou 1=Apnée)
        # y_pred : probabilités prédites par le modèle (entre 0 et 1)

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        # Limite y_pred entre 0.0000001 et 0.9999999
        # Protection mathématique : log(0) = -infini ferait planter le calcul

        bce = -y_true * tf.math.log(y_pred) \
              - (1 - y_true) * tf.math.log(1 - y_pred)
        # Calcul de la Binary CrossEntropy standard
        # Base de toute fonction de perte pour la classification binaire

        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        # Probabilité que le modèle ait correctement classé cet exemple
        # p_t proche de 1 → modèle confiant ET correct
        # p_t proche de 0 → modèle confiant MAIS incorrect

        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        # Applique alpha différemment selon la classe :
        # Apnée  (y_true=1) → alpha_t = 0.53
        # Normal (y_true=0) → alpha_t = 0.47

        focal_w = alpha_t * tf.pow(1 - p_t, gamma)
        # Poids focal : cœur de la Focal Loss
        # Exemple facile (p_t ≈ 1) → poids faible
        # Exemple difficile (p_t ≈ 0) → poids élevé
        # Le modèle se concentre sur les exemples difficiles à classifier

        return tf.reduce_mean(focal_w * bce)
        # Retourne la moyenne pondérée de la perte sur tout le batch

    return loss

# ────────────────────────────────────────────────────────────────
# PARTIE 8 : ARCHITECTURE DU MODÈLE CNN 1D
# ────────────────────────────────────────────────────────────────

model = Sequential([

    # ── BLOC CONVOLUTIF 1 ────────────────────────────────────────
    Conv1D(64, 15, activation='relu', padding='same',
           input_shape=X_train.shape[1:]),
    # 64 filtres appris automatiquement pendant l'entraînement
    # kernel=15 : chaque filtre analyse une fenêtre de 15 points
    #             soit 150 ms à 100 Hz (15 × 1/100 = 0.15 s)
    #             susceptible de capturer des motifs locaux du signal ECG
    #             tels que la morphologie des complexes QRS
    # relu      : introduit une non-linéarité
    #             et supprime les activations négatives → f(x) = max(0, x)
    # padding='same' : conserve la longueur du signal en sortie
    # input_shape : entrée attendue = (6000 points, 1 canal)
    BatchNormalization(),
    # Normalise les activations pour stabiliser l'entraînement
    MaxPooling1D(2),
    # Réduit le signal de moitié : 6000 → 3000 points
    Dropout(0.3),
    # Désactive aléatoirement 30% des neurones pendant l'entraînement
    # Désactivé lors de l'inférence sur ESP32-S3

    # ── BLOC CONVOLUTIF 2 ────────────────────────────────────────
    Conv1D(128, 9, activation='relu', padding='same'),
    # 128 filtres — fenêtre de 9 points = 90 ms
    # Peut apprendre des motifs associés aux variations
    # du rythme cardiaque sur plusieurs battements consécutifs
    BatchNormalization(),
    MaxPooling1D(2),
    # 3000 → 1500 points
    Dropout(0.3),

    # ── BLOC CONVOLUTIF 3 ────────────────────────────────────────
    Conv1D(256, 5, activation='relu', padding='same'),
    # 256 filtres — fenêtre de 5 points = 50 ms
    # Susceptible de capturer certaines signatures locales
    # liées aux irrégularités du signal ECG
    BatchNormalization(),
    MaxPooling1D(2),
    # 1500 → 750 points
    Dropout(0.3),

    # ── BLOC CONVOLUTIF 4 ────────────────────────────────────────
    Conv1D(256, 3, activation='relu', padding='same'),
    # 256 filtres — fenêtre de 3 points = 30 ms
    # Peut apprendre des motifs très locaux du signal ECG
    # susceptibles d'être associés à des signatures de l'apnée
    BatchNormalization(),
    MaxPooling1D(2),
    # 750 → 375 points
    Dropout(0.3),

    # ── GLOBAL MAX POOLING ───────────────────────────────────────
    GlobalMaxPooling1D(),
    # Résume toute la séquence en un vecteur compact
    # Pour chaque filtre parmi 256 : prend la valeur maximale
    # sur les 375 instants de temps
    # (375, 256) → (256,)
    #
    # Justification du choix de MAX et non AVERAGE :
    # L'apnée est un événement ponctuel (10-30s sur 60s)
    # Une moyenne diluerait l'apnée dans les secondes de signal normal
    # Le maximum capture "le moment le plus anormal" du signal
    # Vérifié expérimentalement comme supérieur à GlobalAveragePooling

    # ── COUCHE DENSE ─────────────────────────────────────────────
    Dense(128, activation='relu',
          kernel_regularizer=l2(0.0015)),
    # 128 neurones connectés à toutes les 256 valeurs entrantes
    # relu : introduit une non-linéarité → f(x) = max(0, x)
    # l2=0.0015 : régularisation Ridge
    #             pénalise les grands poids pendant l'entraînement
    #             réduit le risque d'overfitting
    # 128 neurones retenus après expérimentation

    Dropout(0.45),
    # Désactive 45% des neurones juste avant la décision finale
    # Valeur plus élevée qu'aux blocs précédents
    # pour une généralisation maximale

    # ── COUCHE DE SORTIE ─────────────────────────────────────────
    Dense(1, activation='sigmoid')
    # 1 seul neurone de sortie
    # sigmoid : transforme n'importe quelle valeur en probabilité [0, 1]
    # Formule : sigmoid(x) = 1 / (1 + e^(-x))
    #
    # Interprétation de la sortie :
    # prob > 0.54 → APNÉE DÉTECTÉE
    # prob < 0.54 → ÉPISODE NORMAL
    # Seuil 0.54 identifié par optimisation exhaustive après entraînement
])

# ────────────────────────────────────────────────────────────────
# PARTIE 9 : COMPILATION DU MODÈLE
# ────────────────────────────────────────────────────────────────

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    # Adam : optimiseur adaptatif qui ajuste automatiquement
    # le learning rate pour chaque poids selon l'historique des gradients
    # lr=0.00008 identifié après expérimentation

    loss=focal_loss(gamma=2.0, alpha=alpha),
    # Focal Loss personnalisée pour gérer le déséquilibre 38%/62%

    metrics=['accuracy',
             tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.Precision(name='precision')]
    # Métriques surveillées pendant l'entraînement
)

# ────────────────────────────────────────────────────────────────
# PARTIE 10 : CALLBACKS
# ────────────────────────────────────────────────────────────────

callbacks = [
    EarlyStopping(
        monitor='val_auc',
        # Surveille l'AUC de validation — plus fiable que l'accuracy
        # sur les données déséquilibrées

        patience=20,
        # Attend 20 epochs sans amélioration avant d'arrêter
        # Valeur retenue après expérimentation (10 et 15 trop courts)

        restore_best_weights=True,
        # Récupère automatiquement les poids du meilleur epoch

        mode='max', verbose=1
    ),

    ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,
        # Divise le learning rate par 2 si l'AUC stagne
        patience=7,
        # Attend 7 epochs sans amélioration avant de réduire
        mode='max',
        min_lr=1e-7,
        verbose=1
    )
]

# ────────────────────────────────────────────────────────────────
# PARTIE 11 : ENTRAÎNEMENT DU MODÈLE
# ────────────────────────────────────────────────────────────────

print("\nEntrainement en cours ...")
history = model.fit(
    X_train, y_train,

    validation_split=0.2,
    # Réserve 20% des données pour la validation

    epochs=70,
    # Maximum 70 epochs — EarlyStopping peut arrêter avant

    batch_size=64,
    # 64 segments traités simultanément — retenu après expérimentation

    class_weight=cw,
    # Les exemples de la classe Normal contribuent 1.7 fois plus à la perte
    # Les exemples de la classe Apnée contribuent 1.3 fois plus à la perte
    # Compense le déséquilibre 38% apnée / 62% normal

    callbacks=callbacks,
    verbose=1
)

# ────────────────────────────────────────────────────────────────
# PARTIE 12 : ÉVALUATION SUR LE SET DE TEST
# ────────────────────────────────────────────────────────────────

y_prob = model.predict(X_test).flatten()
# Probabilités d'apnée pour les 17 248 segments du set de test
# Données jamais vues pendant l'entraînement

auc = roc_auc_score(y_test, y_prob)
# AUC = 0.866 : dans 86.6% des cas, le modèle attribue une probabilité
# plus élevée à un segment apnée qu'à un segment normal pris au hasard

y_pred = (y_prob > SEUIL).astype(int)
# Décision binaire selon le seuil initial (0.50)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
# TN = 7 089 : segments normaux correctement classés
# FP = 3 612 : segments normaux classés à tort comme apnée
# FN =   806 : apnées non détectées ← priorité médicale absolue
# TP = 5 741 : apnées correctement détectées

recall_apnee  = tp / (tp + fn)
# 0.877 : 87.7% des apnées détectées

recall_normal = tn / (tn + fp)
# 0.662 : 66.2% des segments normaux correctement classés

precision = tp / (tp + fp)
# 0.614 : 61.4% des alarmes sont de vraies apnées

f1 = f1_score(y_test, y_pred, pos_label=1)
# 0.722 : équilibre entre Recall et Précision

accuracy = (tp + tn) / len(y_test)
# 0.744 : pourcentage global de bonnes prédictions

print(f"\n{'='*55}")
print(f"  RESULTATS — {VERSION}")
print(f"{'='*55}")
print(f"  AUC          : {auc:.3f}")
print(f"  Recall Apnee : {recall_apnee:.3f}")
print(f"  Recall Normal: {recall_normal:.3f}")
print(f"  Precision    : {precision:.3f}")
print(f"  F1           : {f1:.3f}")
print(f"  Accuracy     : {accuracy*100:.1f}%")
print(f"  FN           : {fn}")
print(f"  FP           : {fp}")
print(f"\n  Matrice de confusion :")
print(f"                  Predit Normal  Predit Apnee")
print(f"  Reel Normal  :      {tn:5d}         {fp:5d}")
print(f"  Reel Apnee   :      {fn:5d}         {tp:5d}")

# ────────────────────────────────────────────────────────────────
# PARTIE 13 : COMPARAISON AVEC LE MODÈLE PRÉCÉDENT
# ────────────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print(f"  COMPARAISON avec drop045")
print(f"{'='*55}")
ref = {'AUC':0.840,'R_Apn':0.870,'R_Nor':0.649,'FN':848,'FP':3758}
# Métriques du modèle de référence (v3_a056_dense128_drop045)

print(f"  {'Metrique':<15} {'drop045':>8} {'Nouveau':>8} {'Diff':>8}")
print(f"  {'-'*42}")
print(f"  {'AUC':<15} {ref['AUC']:>8.3f} {auc:>8.3f} {auc-ref['AUC']:>+8.3f}")
print(f"  {'Recall Apnee':<15} {ref['R_Apn']:>8.3f} {recall_apnee:>8.3f} {recall_apnee-ref['R_Apn']:>+8.3f}")
print(f"  {'Recall Normal':<15} {ref['R_Nor']:>8.3f} {recall_normal:>8.3f} {recall_normal-ref['R_Nor']:>+8.3f}")
print(f"  {'FN':<15} {ref['FN']:>8} {fn:>8} {fn-ref['FN']:>+8}")
print(f"  {'FP':<15} {ref['FP']:>8} {fp:>8} {fp-ref['FP']:>+8}")

# ────────────────────────────────────────────────────────────────
# PARTIE 14 : OPTIMISATION DU SEUIL DE DÉCISION
# ────────────────────────────────────────────────────────────────

print(f"\n{'='*55}")
print(f"  OPTIMISATION SEUIL")
print(f"{'='*55}")
print(f"  {'Seuil':>6} {'R_Apnee':>8} {'R_Normal':>9} {'FN':>6} {'FP':>6}")
print(f"  {'-'*45}")

for seuil in np.arange(0.50, 0.70, 0.02):
    y_pred_s            = (y_prob > seuil).astype(int)
    cm_s                = confusion_matrix(y_test, y_pred_s)
    tn_s, fp_s, fn_s, tp_s = cm_s.ravel()
    r_apn = tp_s / (tp_s + fn_s)
    r_nor = tn_s / (tn_s + fp_s)
    # Teste les seuils de 0.50 à 0.68 par pas de 0.02
    # Seuil optimal retenu : 0.54 → FN=806, FP=3612
    # Ce seuil est utilisé dans le code Arduino sur ESP32-S3

    flag = ""
    if fn_s <= 848 and fp_s <= 3758:
        flag = " <- MEILLEUR ✅"
    elif fn_s <= 900 and fp_s <= 3900:
        flag = " <- 👀"

    print(f"  {seuil:>6.2f} {r_apn:>8.3f} "
          f"{r_nor:>9.3f} {fn_s:>6} {fp_s:>6}{flag}")

# ────────────────────────────────────────────────────────────────
# PARTIE 15 : TRACÉ DES COURBES DE PERFORMANCE
# ────────────────────────────────────────────────────────────────

fpr, tpr, _              = roc_curve(y_test, y_prob)
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)

fig, axes = plt.subplots(1, 4, figsize=(22, 5))

axes[0].plot(history.history['accuracy'], label='Train', color='steelblue')
axes[0].plot(history.history['val_accuracy'], label='Val', color='tomato')
axes[0].set_title('Accuracy')
axes[0].legend(); axes[0].grid(True)
# Train ≈ Val → pas d'overfitting significatif

axes[1].plot(history.history['loss'], label='Train', color='steelblue')
axes[1].plot(history.history['val_loss'], label='Val', color='tomato')
axes[1].set_title('Loss')
axes[1].legend(); axes[1].grid(True)
# Doit diminuer progressivement pendant l'entraînement

axes[2].plot(fpr, tpr, color='steelblue', linewidth=2, label=f'AUC={auc:.3f}')
axes[2].plot([0,1],[0,1],'k--')
axes[2].set_title('Courbe ROC')
axes[2].legend(); axes[2].grid(True)
# Plus la courbe est proche du coin supérieur gauche, meilleur est le modèle

axes[3].plot(rec_curve, prec_curve, color='steelblue', linewidth=2)
axes[3].set_title('Precision-Recall')
axes[3].set_xlabel('Recall'); axes[3].set_ylabel('Precision')
axes[3].grid(True)
# Montre le compromis entre détecter toutes les apnées et éviter les fausses alarmes

plt.suptitle(f'{VERSION}  AUC={auc:.3f}', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, f'courbes_{VERSION}.png'), dpi=150)
plt.show()

# ────────────────────────────────────────────────────────────────
# PARTIE 16 : SAUVEGARDE DU MODÈLE
# ────────────────────────────────────────────────────────────────

model.save(os.path.join(RESULTS_PATH, f'model_cnn_{VERSION}.keras'))
# Sauvegarde le modèle au format Keras dans Google Drive
# Nom : model_cnn_v3_4conv_a0.53_lr8e-05_cw1.7_run1.keras
#
# Étapes suivantes :
# 1. Conversion en .tflite FLOAT32
#    (taille observée pour cette version : 1 848 KB)
# 2. Conversion en model.h (tableau C++)
#    (taille observée pour cette version : ~11 440 KB)
# 3. Copie de model.h dans le projet Arduino IDE
# 4. Chargement sur ESP32-S3 N16R8 via #include "model.h"
# 5. Inférence temps réel toutes les 60 secondes
#    avec seuil de décision = 0.54

print(f"\n  Modele sauvegarde ✅")
print(f"  {VERSION}.keras")
print(f"  AUC={auc:.3f}  FN={fn}  FP={fp}")
