# Weaken Multiplier 
Weaken = 0
Weaken_Multiplier = 100 - Weaken

# Universal DMG Reduction Multiplier
DMG_Reduction = [0/100]
arr_DMG_Reduction = [1 - DMG_Reductionx for DMG_Reductionx in DMG_Reduction]
Universal_DMG_Reduction_Multiplier = 100
for arr_DMG_Reductiony in arr_DMG_Reduction:
    Universal_DMG_Reduction_Multiplier = arr_DMG_Reductiony * Universal_DMG_Reduction_Multiplier

# DMG Taken Multiplier
Elemental_DMG_Taken = 0
All_Type_DMG_Taken = 0
DMG_Taken_Multiplier = 100 + Elemental_DMG_Taken + All_Type_DMG_Taken

# RES Multiplier
True_RES = 20
RES_PEn = 0
RES_Multiplier = 100 - (True_RES - RES_PEn)
if RES_Multiplier < -100:
    RES_Multiplier = -100
if RES_Multiplier > 90:
    RES_Multiplier = 90

# DEF Multiplier
Base_DEF = 1150
True_DEF = 0
DEF_Reduction = 0
DEF_Ignore = 0
Flat_DEF = 0
DEf = Base_DEF * (100 + True_DEF - (DEF_Reduction + DEF_Ignore))/100 + Flat_DEF
if DEf < 0:
    DEf = 0

Attacker_Level = 90
DEF_Multiplier = 100 - (DEf / (DEf + 200 + 10 * Attacker_Level))*100

# DMG Multiplier
Elemental_DMG = 100
All_Type_DMG = 50
DoT_DMG = 0
Other_DMG = 0
DMG_Multiplier = 100 + Elemental_DMG + All_Type_DMG + DoT_DMG + Other_DMG

# Base DMG
Skill_Multiplier = 2
Extra_Multiplier = 6.72
Scaling_Attribute = 10000
Extra_DMG = 0
Base_DMG = (Skill_Multiplier + Extra_Multiplier) * Scaling_Attribute + Extra_DMG

# Outgoing DMG
Outgoing_DMG = Base_DMG * (DMG_Multiplier/100) * (DEF_Multiplier/100) * (RES_Multiplier/100) * (DMG_Taken_Multiplier/100) * (Universal_DMG_Reduction_Multiplier/100) * (Weaken_Multiplier/100)
print(f"Base_DMG: {Base_DMG}")
print(f"DMG_Multiplier: {DMG_Multiplier}")
print(f"DEF_Multiplier: {DEF_Multiplier}")
print(f"RES_Multiplier: {RES_Multiplier}")
print(f"DMG_Taken_Multiplier: {DMG_Taken_Multiplier}")
print(f"Universal_DMG_Reduction_Multiplier: {Universal_DMG_Reduction_Multiplier}")
print(f"Weaken_Multiplier: {Weaken_Multiplier}")
print(Outgoing_DMG*3*1.4)