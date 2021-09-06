import rdkit
from rdkit import Chem
from rdkit.Chem import Lipinski
import rdkit.Chem.rdMolDescriptors as Descriptor
import pandas as pd
import math
from rdkit.Chem import Descriptors
print(rdkit.__version__)


def BBB_Descriptor(mol):
    """
    Calculate the descriptors for Gupa(2019) BBB Score

    Usage: BBB_Descriptor(mol)
    The above mol should be a RDKit molecule. And return a list including HA, MWHBN, ARO_R and TPSA.

    Return: [HA, MWHBN, ARO_R, TPSA]

    """

    # Calculate NWHBN
    nHBA = Descriptor.CalcNumHBA(mol)
    nHBD = Descriptor.CalcNumHBD(mol)
    HBN = nHBA + nHBD
    MW = round(Descriptor.CalcExactMolWt(mol, onlyHeavy=False), 2)
    MWHBN = round(HBN / (MW ** 0.5), 2)

    # heavy atoms
    HA = Lipinski.HeavyAtomCount(mol)

    # Calculate the number of aromatic rings
    aroR = Descriptor.CalcNumAromaticRings(mol)

    # TPSA
    tpsa = Descriptor.CalcTPSA(mol)

    BBB_DESC = [MW, nHBA, nHBD, HBN, MWHBN, HA, aroR, tpsa]
    return (BBB_DESC)


def re_bbbscore(Aro_R, HA, HBN, MW, TPSA, pka):
    """Calculate BBB SCORE"""

    # Calculate P value for Aromatic Rings
    if Aro_R == 0:
        P_ARO_R = 0.336376

    elif Aro_R == 1:
        P_ARO_R = 0.816016

    elif Aro_R == 2:
        P_ARO_R = 1

    elif Aro_R == 3:
        P_ARO_R = 0.691115

    elif Aro_R == 4:
        P_ARO_R = 0.199399

    elif Aro_R > 4:
        P_ARO_R = 0

    # Calculate P value for HA
    if HA > 5 and HA <= 45:
        P_HA = (0.0000443 * (HA ** 3) - 0.004556 * (HA ** 2) + 0.12775 * HA - 0.463) / 0.624231
    else:
        P_HA = 0

    # Calculate P value for MWHBN
    MWHBN = round(HBN / (MW ** 0.5), 2)
    if MWHBN > 0.05 and MWHBN <= 0.45:
        P_MWHBN = (26.733 * (MWHBN ** 3) - 31.495 * (MWHBN ** 2) + 9.5202 * MWHBN - 0.1358) / 0.72258
    else:
        P_MWHBN = 0

    # Calculate P value for TPSA
    if TPSA > 0 and TPSA <= 120:
        P_TPSA = (-0.0067 * TPSA + 0.9598) / 0.9598
    else:
        P_TPSA = 0

    # Calculate P value for pKa
    pka = float(pka)
    if math.isnan(pka):
        pka = 8.81
    if pka > 3 and pka <= 11:
        P_PKA = (0.00045068 * (pka ** 4) - 0.016331 * (pka ** 3) + 0.18618 * (
                    pka ** 2) - 0.71043 * pka + 0.8579) / 0.597488
    else:
        P_PKA = 0

    BBB_score = P_ARO_R + P_HA + 1.5 * P_MWHBN + 2 * P_TPSA + 0.5 * P_PKA

    return (round(BBB_score,2))

def bbbscore(mol, pka):
    """Calculate BBB SCORE"""
    BBB_DESC = BBB_Descriptor(mol)
    HA = float(BBB_DESC[5])
    MWHBN = float(BBB_DESC[4])
    Aro_R = float(BBB_DESC[6])
    TPSA = float(BBB_DESC[7])
    BBB_SCORE = []

    # Calculate P value for Aromatic Rings
    if Aro_R == 0:
        P_ARO_R = 0.336376

    elif Aro_R == 1:
        P_ARO_R = 0.816016

    elif Aro_R == 2:
        P_ARO_R = 1

    elif Aro_R == 3:
        P_ARO_R = 0.691115

    elif Aro_R == 4:
        P_ARO_R = 0.199399

    elif Aro_R > 4:
        P_ARO_R = 0

    # Calculate P value for HA
    if HA > 5 and HA <= 45:
        P_HA = (0.0000443 * (HA ** 3) - 0.004556 * (HA ** 2) + 0.12775 * HA - 0.463) / 0.624231
    else:
        P_HA = 0

    # Calculate P value for MWHBN
    if MWHBN > 0.05 and MWHBN <= 0.45:
        P_MWHBN = (26.733 * (MWHBN ** 3) - 31.495 * (MWHBN ** 2) + 9.5202 * MWHBN - 0.1358) / 0.72258
    else:
        P_MWHBN = 0

    # Calculate P value for TPSA
    if TPSA > 0 and TPSA <= 120:
        P_TPSA = (-0.0067 * TPSA + 0.9598) / 0.9598
    else:
        P_TPSA = 0

    # Calculate P value for pKa
    pka = float(pka)
    if math.isnan(pka):
        pka = 8.81
    if pka > 3 and pka <= 11:
        P_PKA = (0.00045068 * (pka ** 4) - 0.016331 * (pka ** 3) + 0.18618 * (
                    pka ** 2) - 0.71043 * pka + 0.8579) / 0.597488
    else:
        P_PKA = 0

    BBB_score = P_ARO_R + P_HA + 1.5 * P_MWHBN + 2 * P_TPSA + 0.5 * P_PKA

    BBB_DESC.append(pka)
    BBB_SCORE = BBB_DESC
    #print(BBB_SCORE)
    BBB_SCORE.append(round(BBB_score, 2))

    return (BBB_SCORE)

# Print the descriptors and BBB score
def BBB_Score_Report(bbb_score):
    #[MW, nHBA, nHBD, HBN, MWHBN, HA, aroR, tpsa, pKa, BBB score]
    print("Number of Aromatic Rings(Aro_R):",bbb_score[6])
    print("Number of Heavy Atoms(HA):",bbb_score[5])
    print("Molecular Weight(MW):",bbb_score[0])
    print("Number of Hydrogen Bond Acceptor (HBA):",bbb_score[1])
    print("Number of Hydrogen Bond Donor(HBD):",bbb_score[2])
    print("MWHBN = HBN/MW^0.5, HBN = HBA + HBD:",bbb_score[4])
    print("Topological Polar Surface Area(TPSA):",bbb_score[7])
    print("pKa:",bbb_score[8])
    print("BBB Score:",bbb_score[9])

