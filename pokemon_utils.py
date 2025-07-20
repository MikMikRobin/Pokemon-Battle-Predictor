"""
Utility functions for Pokémon visualization and data handling.
"""
import os
import requests
import matplotlib.pyplot as plt
from PIL import Image
import io
import pandas as pd
import numpy as np

# Create a sprites directory if it doesn't exist
SPRITES_DIR = "sprites"
os.makedirs(SPRITES_DIR, exist_ok=True)

# Load the Pokémon data to map IDs correctly
def load_pokemon_data():
    """Load Pokémon data from CSV to map IDs correctly"""
    try:
        pokemon_df = pd.read_csv("pokemon.csv")
        return pokemon_df
    except Exception as e:
        print(f"Error loading Pokémon data: {e}")
        return None

# Map our Pokémon ID to the National Pokédex ID
def map_to_national_dex_id(pokemon_id):
    """Map our dataset Pokémon ID to the National Pokédex ID for sprite fetching"""
    # Create a complete mapping of our dataset IDs to National Pokédex IDs
    # This is based on the pokemon.csv file
    id_mapping = {
        # Gen 1 (1-166)
        1: 1,    # Bulbasaur
        2: 2,    # Ivysaur
        3: 3,    # Venusaur
        4: 3,    # Mega Venusaur
        5: 4,    # Charmander
        6: 5,    # Charmeleon
        7: 6,    # Charizard
        8: 6,    # Mega Charizard X
        9: 6,    # Mega Charizard Y
        10: 7,   # Squirtle
        11: 8,   # Wartortle
        12: 9,   # Blastoise
        13: 9,   # Mega Blastoise
        14: 10,  # Caterpie
        15: 11,  # Metapod
        16: 12,  # Butterfree
        17: 13,  # Weedle
        18: 14,  # Kakuna
        19: 15,  # Beedrill
        20: 15,  # Mega Beedrill
        21: 16,  # Pidgey
        22: 17,  # Pidgeotto
        23: 18,  # Pidgeot
        24: 18,  # Mega Pidgeot
        25: 19,  # Rattata
        26: 20,  # Raticate
        27: 21,  # Spearow
        28: 22,  # Fearow
        29: 23,  # Ekans
        30: 24,  # Arbok
        31: 25,  # Pikachu
        32: 26,  # Raichu
        33: 27,  # Sandshrew
        34: 28,  # Sandslash
        35: 29,  # Nidoran♀
        36: 30,  # Nidorina
        37: 31,  # Nidoqueen
        38: 32,  # Nidoran♂
        39: 33,  # Nidorino
        40: 34,  # Nidoking
        41: 35,  # Clefairy
        42: 36,  # Clefable
        43: 37,  # Vulpix
        44: 38,  # Ninetales
        45: 39,  # Jigglypuff
        46: 40,  # Wigglytuff
        47: 41,  # Zubat
        48: 42,  # Golbat
        49: 43,  # Oddish
        50: 44,  # Gloom
        51: 45,  # Vileplume
        52: 46,  # Paras
        53: 47,  # Parasect
        54: 48,  # Venonat
        55: 49,  # Venomoth
        56: 50,  # Diglett
        57: 51,  # Dugtrio
        58: 52,  # Meowth
        59: 53,  # Persian
        60: 54,  # Psyduck
        61: 55,  # Golduck
        62: 56,  # Mankey
        63: 57,  # Primeape
        64: 58,  # Growlithe
        65: 59,  # Arcanine
        66: 60,  # Poliwag
        67: 61,  # Poliwhirl
        68: 62,  # Poliwrath
        69: 63,  # Abra
        70: 64,  # Kadabra
        71: 65,  # Alakazam
        72: 65,  # Mega Alakazam
        73: 66,  # Machop
        74: 67,  # Machoke
        75: 68,  # Machamp
        76: 69,  # Bellsprout
        77: 70,  # Weepinbell
        78: 71,  # Victreebel
        79: 72,  # Tentacool
        80: 73,  # Tentacruel
        81: 74,  # Geodude
        82: 75,  # Graveler
        83: 76,  # Golem
        84: 77,  # Ponyta
        85: 78,  # Rapidash
        86: 79,  # Slowpoke
        87: 80,  # Slowbro
        88: 80,  # Mega Slowbro
        89: 81,  # Magnemite
        90: 82,  # Magneton
        91: 83,  # Farfetch'd
        92: 84,  # Doduo
        93: 85,  # Dodrio
        94: 86,  # Seel
        95: 87,  # Dewgong
        96: 88,  # Grimer
        97: 89,  # Muk
        98: 90,  # Shellder
        99: 91,  # Cloyster
        100: 92,  # Gastly
        101: 93,  # Haunter
        102: 94,  # Gengar
        103: 94,  # Mega Gengar
        104: 95,  # Onix
        105: 96,  # Drowzee
        106: 97,  # Hypno
        107: 98,  # Krabby
        108: 99,  # Kingler
        109: 100, # Voltorb
        110: 101, # Electrode
        111: 102, # Exeggcute
        112: 103, # Exeggutor
        113: 104, # Cubone
        114: 105, # Marowak
        115: 106, # Hitmonlee
        116: 107, # Hitmonchan
        117: 108, # Lickitung
        118: 109, # Koffing
        119: 110, # Weezing
        120: 111, # Rhyhorn
        121: 112, # Rhydon
        122: 113, # Chansey
        123: 114, # Tangela
        124: 115, # Kangaskhan
        125: 115, # Mega Kangaskhan
        126: 116, # Horsea
        127: 117, # Seadra
        128: 118, # Goldeen
        129: 119, # Seaking
        130: 120, # Staryu
        131: 121, # Starmie
        132: 122, # Mr. Mime
        133: 123, # Scyther
        134: 124, # Jynx
        135: 125, # Electabuzz
        136: 126, # Magmar
        137: 127, # Pinsir
        138: 127, # Mega Pinsir
        139: 128, # Tauros
        140: 129, # Magikarp
        141: 130, # Gyarados
        142: 130, # Mega Gyarados
        143: 131, # Lapras
        144: 132, # Ditto
        145: 133, # Eevee
        146: 134, # Vaporeon
        147: 135, # Jolteon
        148: 136, # Flareon
        149: 137, # Porygon
        150: 138, # Omanyte
        151: 139, # Omastar
        152: 140, # Kabuto
        153: 141, # Kabutops
        154: 142, # Aerodactyl
        155: 142, # Mega Aerodactyl
        156: 143, # Snorlax
        157: 144, # Articuno
        158: 145, # Zapdos
        159: 146, # Moltres
        160: 147, # Dratini
        161: 148, # Dragonair
        162: 149, # Dragonite
        163: 150, # Mewtwo
        164: 150, # Mega Mewtwo X
        165: 150, # Mega Mewtwo Y
        166: 151, # Mew
        
        # Gen 2 (167-272)
        167: 152, # Chikorita
        168: 153, # Bayleef
        169: 154, # Meganium
        170: 155, # Cyndaquil
        171: 156, # Quilava
        172: 157, # Typhlosion
        173: 158, # Totodile
        174: 159, # Croconaw
        175: 160, # Feraligatr
        176: 161, # Sentret
        177: 162, # Furret
        178: 163, # Hoothoot
        179: 164, # Noctowl
        180: 165, # Ledyba
        181: 166, # Ledian
        182: 167, # Spinarak
        183: 168, # Ariados
        184: 169, # Crobat
        185: 170, # Chinchou
        186: 171, # Lanturn
        187: 172, # Pichu
        188: 173, # Cleffa
        189: 174, # Igglybuff
        190: 175, # Togepi
        191: 176, # Togetic
        192: 177, # Natu
        193: 178, # Xatu
        194: 179, # Mareep
        195: 180, # Flaaffy
        196: 181, # Ampharos
        197: 181, # Mega Ampharos
        198: 182, # Bellossom
        199: 183, # Marill
        200: 184, # Azumarill
        201: 185, # Sudowoodo
        202: 186, # Politoed
        203: 187, # Hoppip
        204: 188, # Skiploom
        205: 189, # Jumpluff
        206: 190, # Aipom
        207: 191, # Sunkern
        208: 192, # Sunflora
        209: 193, # Yanma
        210: 194, # Wooper
        211: 195, # Quagsire
        212: 196, # Espeon
        213: 197, # Umbreon
        214: 198, # Murkrow
        215: 199, # Slowking
        216: 200, # Misdreavus
        217: 201, # Unown
        218: 202, # Wobbuffet
        219: 203, # Girafarig
        220: 204, # Pineco
        221: 205, # Forretress
        222: 206, # Dunsparce
        223: 207, # Gligar
        224: 208, # Steelix
        225: 208, # Mega Steelix
        226: 209, # Snubbull
        227: 210, # Granbull
        228: 211, # Qwilfish
        229: 212, # Scizor
        230: 212, # Mega Scizor
        231: 213, # Shuckle
        232: 214, # Heracross
        233: 214, # Mega Heracross
        234: 215, # Sneasel
        235: 216, # Teddiursa
        236: 217, # Ursaring
        237: 218, # Slugma
        238: 219, # Magcargo
        239: 220, # Swinub
        240: 221, # Piloswine
        241: 222, # Corsola
        242: 223, # Remoraid
        243: 224, # Octillery
        244: 225, # Delibird
        245: 226, # Mantine
        246: 227, # Skarmory
        247: 228, # Houndour
        248: 229, # Houndoom
        249: 229, # Mega Houndoom
        250: 230, # Kingdra
        251: 231, # Phanpy
        252: 232, # Donphan
        253: 233, # Porygon2
        254: 234, # Stantler
        255: 235, # Smeargle
        256: 236, # Tyrogue
        257: 237, # Hitmontop
        258: 238, # Smoochum
        259: 239, # Elekid
        260: 240, # Magby
        261: 241, # Miltank
        262: 242, # Blissey
        263: 243, # Raikou
        264: 244, # Entei
        265: 245, # Suicune
        266: 246, # Larvitar
        267: 247, # Pupitar
        268: 248, # Tyranitar
        269: 248, # Mega Tyranitar
        270: 249, # Lugia
        271: 250, # Ho-oh
        272: 251, # Celebi
        
        # Gen 3 (273-414)
        273: 252, # Treecko
        274: 253, # Grovyle
        275: 254, # Sceptile
        276: 254, # Mega Sceptile
        277: 255, # Torchic
        278: 256, # Combusken
        279: 257, # Blaziken
        280: 257, # Mega Blaziken
        281: 258, # Mudkip
        282: 259, # Marshtomp
        283: 260, # Swampert
        284: 260, # Mega Swampert
        285: 261, # Poochyena
        286: 262, # Mightyena
        287: 263, # Zigzagoon
        288: 264, # Linoone
        289: 265, # Wurmple
        290: 266, # Silcoon
        291: 267, # Beautifly
        292: 268, # Cascoon
        293: 269, # Dustox
        294: 270, # Lotad
        295: 271, # Lombre
        296: 272, # Ludicolo
        297: 273, # Seedot
        298: 274, # Nuzleaf
        299: 275, # Shiftry
        300: 276, # Taillow
        301: 277, # Swellow
        302: 278, # Wingull
        303: 279, # Pelipper
        304: 280, # Ralts
        305: 281, # Kirlia
        306: 282, # Gardevoir
        307: 282, # Mega Gardevoir
        308: 283, # Surskit
        309: 284, # Masquerain
        310: 285, # Shroomish
        311: 286, # Breloom
        312: 287, # Slakoth
        313: 288, # Vigoroth
        314: 289, # Slaking
        315: 290, # Nincada
        316: 291, # Ninjask
        317: 292, # Shedinja
        318: 293, # Whismur
        319: 294, # Loudred
        320: 295, # Exploud
        321: 296, # Makuhita
        322: 297, # Hariyama
        323: 298, # Azurill
        324: 299, # Nosepass
        325: 300, # Skitty
        326: 301, # Delcatty
        327: 302, # Sableye
        328: 302, # Mega Sableye
        329: 303, # Mawile
        330: 303, # Mega Mawile
        331: 304, # Aron
        332: 305, # Lairon
        333: 306, # Aggron
        334: 306, # Mega Aggron
        335: 307, # Meditite
        336: 308, # Medicham
        337: 308, # Mega Medicham
        338: 309, # Electrike
        339: 310, # Manectric
        340: 310, # Mega Manectric
        341: 311, # Plusle
        342: 312, # Minun
        343: 313, # Volbeat
        344: 314, # Illumise
        345: 315, # Roselia
        346: 316, # Gulpin
        347: 317, # Swalot
        348: 318, # Carvanha
        349: 319, # Sharpedo
        350: 319, # Mega Sharpedo
        351: 320, # Wailmer
        352: 321, # Wailord
        353: 322, # Numel
        354: 323, # Camerupt
        355: 323, # Mega Camerupt
        356: 324, # Torkoal
        357: 325, # Spoink
        358: 326, # Grumpig
        359: 327, # Spinda
        360: 328, # Trapinch
        361: 329, # Vibrava
        362: 330, # Flygon
        363: 331, # Cacnea
        364: 332, # Cacturne
        365: 333, # Swablu
        366: 334, # Altaria
        367: 334, # Mega Altaria
        368: 335, # Zangoose
        369: 336, # Seviper
        370: 337, # Lunatone
        371: 338, # Solrock
        372: 339, # Barboach
        373: 340, # Whiscash
        374: 341, # Corphish
        375: 342, # Crawdaunt
        376: 343, # Baltoy
        377: 344, # Claydol
        378: 345, # Lileep
        379: 346, # Cradily
        380: 347, # Anorith
        381: 348, # Armaldo
        382: 349, # Feebas
        383: 350, # Milotic
        384: 351, # Castform
        385: 352, # Kecleon
        386: 353, # Shuppet
        387: 354, # Banette
        388: 354, # Mega Banette
        389: 355, # Duskull
        390: 356, # Dusclops
        391: 357, # Tropius
        392: 358, # Chimecho
        393: 359, # Absol
        394: 359, # Mega Absol
        395: 360, # Wynaut
        396: 361, # Snorunt
        397: 362, # Glalie
        398: 362, # Mega Glalie
        399: 363, # Spheal
        400: 364, # Sealeo
        401: 365, # Walrein
        402: 366, # Clamperl
        403: 367, # Huntail
        404: 368, # Gorebyss
        405: 369, # Relicanth
        406: 370, # Luvdisc
        407: 371, # Bagon
        408: 372, # Shelgon
        409: 373, # Salamence
        410: 373, # Mega Salamence
        411: 374, # Beldum
        412: 375, # Metang
        413: 376, # Metagross
        414: 376, # Mega Metagross
        415: 377, # Regirock
        416: 378, # Regice
        417: 379, # Registeel
        418: 380, # Latias
        419: 380, # Mega Latias
        420: 381, # Latios
        421: 381, # Mega Latios
        422: 382, # Kyogre
        423: 382, # Primal Kyogre
        424: 383, # Groudon
        425: 383, # Primal Groudon
        426: 384, # Rayquaza
        427: 384, # Mega Rayquaza
        428: 385, # Jirachi
        429: 386, # Deoxys Normal Forme
        430: 386, # Deoxys Attack Forme
        431: 386, # Deoxys Defense Forme
        432: 386, # Deoxys Speed Forme
        
        # Gen 4 (433-553)
        433: 387, # Turtwig
        434: 388, # Grotle
        435: 389, # Torterra
        436: 390, # Chimchar
        437: 391, # Monferno
        438: 392, # Infernape
        439: 393, # Piplup
        440: 394, # Prinplup
        441: 395, # Empoleon
        442: 396, # Starly
        443: 397, # Staravia
        444: 398, # Staraptor
        445: 399, # Bidoof
        446: 400, # Bibarel
        447: 401, # Kricketot
        448: 402, # Kricketune
        449: 403, # Shinx
        450: 404, # Luxio
        451: 405, # Luxray
        452: 406, # Budew
        453: 407, # Roserade
        454: 408, # Cranidos
        455: 409, # Rampardos
        456: 410, # Shieldon
        457: 411, # Bastiodon
        458: 412, # Burmy
        459: 413, # Wormadam Plant Cloak
        460: 413, # Wormadam Sandy Cloak
        461: 413, # Wormadam Trash Cloak
        462: 414, # Mothim
        463: 415, # Combee
        464: 416, # Vespiquen
        465: 417, # Pachirisu
        466: 418, # Buizel
        467: 419, # Floatzel
        468: 420, # Cherubi
        469: 421, # Cherrim
        470: 422, # Shellos
        471: 423, # Gastrodon
        472: 424, # Ambipom
        473: 425, # Drifloon
        474: 426, # Drifblim
        475: 427, # Buneary
        476: 428, # Lopunny
        477: 428, # Mega Lopunny
        478: 429, # Mismagius
        479: 430, # Honchkrow
        480: 431, # Glameow
        481: 432, # Purugly
        482: 433, # Chingling
        483: 434, # Stunky
        484: 435, # Skuntank
        485: 436, # Bronzor
        486: 437, # Bronzong
        487: 438, # Bonsly
        488: 439, # Mime Jr.
        489: 440, # Happiny
        490: 441, # Chatot
        491: 442, # Spiritomb
        492: 443, # Gible
        493: 444, # Gabite
        494: 445, # Garchomp
        495: 445, # Mega Garchomp
        496: 446, # Munchlax
        497: 447, # Riolu
        498: 448, # Lucario
        499: 448, # Mega Lucario
        500: 449, # Hippopotas
        501: 450, # Hippowdon
        502: 451, # Skorupi
        503: 452, # Drapion
        504: 453, # Croagunk
        505: 454, # Toxicroak
        506: 455, # Carnivine
        507: 456, # Finneon
        508: 457, # Lumineon
        509: 458, # Mantyke
        510: 459, # Snover
        511: 460, # Abomasnow
        512: 460, # Mega Abomasnow
        513: 461, # Weavile
        514: 462, # Magnezone
        515: 463, # Lickilicky
        516: 464, # Rhyperior
        517: 465, # Tangrowth
        518: 466, # Electivire
        519: 467, # Magmortar
        520: 468, # Togekiss
        521: 469, # Yanmega
        522: 470, # Leafeon
        523: 471, # Glaceon
        524: 472, # Gliscor
        525: 473, # Mamoswine
        526: 474, # Porygon-Z
        527: 475, # Gallade
        528: 475, # Mega Gallade
        529: 476, # Probopass
        530: 477, # Dusknoir
        531: 478, # Froslass
        532: 479, # Rotom
        533: 479, # Heat Rotom
        534: 479, # Wash Rotom
        535: 479, # Frost Rotom
        536: 479, # Fan Rotom
        537: 479, # Mow Rotom
        538: 480, # Uxie
        539: 481, # Mesprit
        540: 482, # Azelf
        541: 483, # Dialga
        542: 484, # Palkia
        543: 485, # Heatran
        544: 486, # Regigigas
        545: 487, # Giratina Altered Forme
        546: 487, # Giratina Origin Forme
        547: 488, # Cresselia
        548: 489, # Phione
        549: 490, # Manaphy
        550: 491, # Darkrai
        551: 492, # Shaymin Land Forme
        552: 492, # Shaymin Sky Forme
        553: 493, # Arceus
        
        # Gen 5 (554-649)
        554: 494, # Victini
        555: 495, # Snivy
        556: 496, # Servine
        557: 497, # Serperior
        558: 498, # Tepig
        559: 499, # Pignite
        560: 500, # Emboar
        561: 501, # Oshawott
        562: 502, # Dewott
        563: 503, # Samurott
        564: 504, # Patrat
        565: 505, # Watchog
        566: 506, # Lillipup
        567: 507, # Herdier
        568: 508, # Stoutland
        569: 509, # Purrloin
        570: 510, # Liepard
        571: 511, # Pansage
        572: 512, # Simisage
        573: 513, # Pansear
        574: 514, # Simisear
        575: 515, # Panpour
        576: 516, # Simipour
        577: 517, # Munna
        578: 518, # Musharna
        579: 519, # Pidove
        580: 520, # Tranquill
        581: 521, # Unfezant
        582: 522, # Blitzle
        583: 523, # Zebstrika
        584: 524, # Roggenrola
        585: 525, # Boldore
        586: 526, # Gigalith
        587: 527, # Woobat
        588: 528, # Swoobat
        589: 529, # Drilbur
        590: 530, # Excadrill
        591: 531, # Audino
        592: 531, # Mega Audino
        593: 532, # Timburr
        594: 533, # Gurdurr
        595: 534, # Conkeldurr
        596: 535, # Tympole
        597: 536, # Palpitoad
        598: 537, # Seismitoad
        599: 538, # Throh
        600: 539  # Sawk
    }
    
    # Check if the ID is in our mapping
    if pokemon_id in id_mapping:
        return id_mapping[pokemon_id]
    
    # If not in mapping, try to get it from the name
    pokemon_df = load_pokemon_data()
    if pokemon_df is None:
        return pokemon_id  # Fallback to using the provided ID
    
    try:
        # Find the row with our ID
        pokemon_row = pokemon_df[pokemon_df['#'] == pokemon_id]
        if len(pokemon_row) == 0:
            return pokemon_id  # Fallback if not found
        
        # Get the name and clean it
        name = pokemon_row['Name'].values[0]
        
        # Create a mapping of Pokémon names to National Dex IDs
        # This is a hardcoded mapping for common Pokémon that might have issues
        name_to_national_dex = {
            "Abomasnow": 460,
            "Aegislash Blade Forme": 681,
            "Aegislash Shield Forme": 681,
            "Meowstic Male": 678,
            "Meowstic Female": 678,
            "Giratina Altered Forme": 487,
            "Giratina Origin Forme": 487,
            "Deoxys Normal Forme": 386,
            "Deoxys Attack Forme": 386,
            "Deoxys Defense Forme": 386,
            "Deoxys Speed Forme": 386,
            "Wormadam Plant Cloak": 413,
            "Wormadam Sandy Cloak": 413,
            "Wormadam Trash Cloak": 413,
            "Shaymin Land Forme": 492,
            "Shaymin Sky Forme": 492,
            "Tornadus Incarnate Forme": 641,
            "Tornadus Therian Forme": 641,
            "Thundurus Incarnate Forme": 642,
            "Thundurus Therian Forme": 642,
            "Landorus Incarnate Forme": 645,
            "Landorus Therian Forme": 645,
            "Kyurem Black Kyurem": 646,
            "Kyurem White Kyurem": 646,
            "Keldeo Ordinary Forme": 647,
            "Keldeo Resolute Forme": 647,
            "Meloetta Aria Forme": 648,
            "Meloetta Pirouette Forme": 648,
            "Darmanitan Standard Mode": 555,
            "Darmanitan Zen Mode": 555,
            "Pumpkaboo Average Size": 710,
            "Pumpkaboo Small Size": 710,
            "Pumpkaboo Large Size": 710,
            "Pumpkaboo Super Size": 710,
            "Gourgeist Average Size": 711,
            "Gourgeist Small Size": 711,
            "Gourgeist Large Size": 711,
            "Gourgeist Super Size": 711,
            "Zygarde Half Forme": 718,
            "Hoopa Confined": 720,
            "Hoopa Unbound": 720
        }
        
        # Add the rest of the IDs (601-800)
        id_mapping.update({
            601: 540, # Sewaddle
            602: 541, # Swadloon
            603: 542, # Leavanny
            604: 543, # Venipede
            605: 544, # Whirlipede
            606: 545, # Scolipede
            607: 546, # Cottonee
            608: 547, # Whimsicott
            609: 548, # Petilil
            610: 549, # Lilligant
            611: 550, # Basculin
            612: 551, # Sandile
            613: 552, # Krokorok
            614: 553, # Krookodile
            615: 554, # Darumaka
            616: 555, # Darmanitan Standard Mode
            617: 555, # Darmanitan Zen Mode
            618: 556, # Maractus
            619: 557, # Dwebble
            620: 558, # Crustle
            621: 559, # Scraggy
            622: 560, # Scrafty
            623: 561, # Sigilyph
            624: 562, # Yamask
            625: 563, # Cofagrigus
            626: 564, # Tirtouga
            627: 565, # Carracosta
            628: 566, # Archen
            629: 567, # Archeops
            630: 568, # Trubbish
            631: 569, # Garbodor
            632: 570, # Zorua
            633: 571, # Zoroark
            634: 572, # Minccino
            635: 573, # Cinccino
            636: 574, # Gothita
            637: 575, # Gothorita
            638: 576, # Gothitelle
            639: 577, # Solosis
            640: 578, # Duosion
            641: 579, # Reuniclus
            642: 580, # Ducklett
            643: 581, # Swanna
            644: 582, # Vanillite
            645: 583, # Vanillish
            646: 584, # Vanilluxe
            647: 585, # Deerling
            648: 586, # Sawsbuck
            649: 587, # Emolga
            650: 588, # Karrablast
            651: 589, # Escavalier
            652: 590, # Foongus
            653: 591, # Amoonguss
            654: 592, # Frillish
            655: 593, # Jellicent
            656: 594, # Alomomola
            657: 595, # Joltik
            658: 596, # Galvantula
            659: 597, # Ferroseed
            660: 598, # Ferrothorn
            661: 599, # Klink
            662: 600, # Klang
            663: 601, # Klinklang
            664: 602, # Tynamo
            665: 603, # Eelektrik
            666: 604, # Eelektross
            667: 605, # Elgyem
            668: 606, # Beheeyem
            669: 607, # Litwick
            670: 608, # Lampent
            671: 609, # Chandelure
            672: 610, # Axew
            673: 611, # Fraxure
            674: 612, # Haxorus
            675: 613, # Cubchoo
            676: 614, # Beartic
            677: 615, # Cryogonal
            678: 616, # Shelmet
            679: 617, # Accelgor
            680: 618, # Stunfisk
            681: 619, # Mienfoo
            682: 620, # Mienshao
            683: 621, # Druddigon
            684: 622, # Golett
            685: 623, # Golurk
            686: 624, # Pawniard
            687: 625, # Bisharp
            688: 626, # Bouffalant
            689: 627, # Rufflet
            690: 628, # Braviary
            691: 629, # Vullaby
            692: 630, # Mandibuzz
            693: 631, # Heatmor
            694: 632, # Durant
            695: 633, # Deino
            696: 634, # Zweilous
            697: 635, # Hydreigon
            698: 636, # Larvesta
            699: 637, # Volcarona
            700: 638, # Cobalion
            701: 639, # Terrakion
            702: 640, # Virizion
            703: 641, # Tornadus Incarnate Forme
            704: 641, # Tornadus Therian Forme
            705: 642, # Thundurus Incarnate Forme
            706: 642, # Thundurus Therian Forme
            707: 643, # Reshiram
            708: 644, # Zekrom
            709: 645, # Landorus Incarnate Forme
            710: 645, # Landorus Therian Forme
            711: 646, # Kyurem
            712: 646, # Kyurem Black Kyurem
            713: 646, # Kyurem White Kyurem
            714: 647, # Keldeo Ordinary Forme
            715: 647, # Keldeo Resolute Forme
            716: 648, # Meloetta Aria Forme
            717: 648, # Meloetta Pirouette Forme
            718: 649, # Genesect
            
            # Gen 6 (719-800)
            719: 650, # Chespin
            720: 651, # Quilladin
            721: 652, # Chesnaught
            722: 653, # Fennekin
            723: 654, # Braixen
            724: 655, # Delphox
            725: 656, # Froakie
            726: 657, # Frogadier
            727: 658, # Greninja
            728: 659, # Bunnelby
            729: 660, # Diggersby
            730: 661, # Fletchling
            731: 662, # Fletchinder
            732: 663, # Talonflame
            733: 664, # Scatterbug
            734: 665, # Spewpa
            735: 666, # Vivillon
            736: 667, # Litleo
            737: 668, # Pyroar
            738: 669, # Flabébé
            739: 670, # Floette
            740: 671, # Florges
            741: 672, # Skiddo
            742: 673, # Gogoat
            743: 674, # Pancham
            744: 675, # Pangoro
            745: 676, # Furfrou
            746: 677, # Espurr
            747: 678, # Meowstic Male
            748: 678, # Meowstic Female
            749: 679, # Honedge
            750: 680, # Doublade
            751: 681, # Aegislash Blade Forme
            752: 681, # Aegislash Shield Forme
            753: 682, # Spritzee
            754: 683, # Aromatisse
            755: 684, # Swirlix
            756: 685, # Slurpuff
            757: 686, # Inkay
            758: 687, # Malamar
            759: 688, # Binacle
            760: 689, # Barbaracle
            761: 690, # Skrelp
            762: 691, # Dragalge
            763: 692, # Clauncher
            764: 693, # Clawitzer
            765: 694, # Helioptile
            766: 695, # Heliolisk
            767: 696, # Tyrunt
            768: 697, # Tyrantrum
            769: 698, # Amaura
            770: 699, # Aurorus
            771: 700, # Sylveon
            772: 701, # Hawlucha
            773: 702, # Dedenne
            774: 703, # Carbink
            775: 704, # Goomy
            776: 705, # Sliggoo
            777: 706, # Goodra
            778: 707, # Klefki
            779: 708, # Phantump
            780: 709, # Trevenant
            781: 710, # Pumpkaboo Average Size
            782: 710, # Pumpkaboo Small Size
            783: 710, # Pumpkaboo Large Size
            784: 710, # Pumpkaboo Super Size
            785: 711, # Gourgeist Average Size
            786: 711, # Gourgeist Small Size
            787: 711, # Gourgeist Large Size
            788: 711, # Gourgeist Super Size
            789: 712, # Bergmite
            790: 713, # Avalugg
            791: 714, # Noibat
            792: 715, # Noivern
            793: 716, # Xerneas
            794: 717, # Yveltal
            795: 718, # Zygarde Half Forme
            796: 719, # Diancie
            797: 719, # Mega Diancie
            798: 720, # Hoopa Confined
            799: 720, # Hoopa Unbound
            800: 721  # Volcanion
        })
        
        # Check if the name is in our mapping
        if name in name_to_national_dex:
            return name_to_national_dex[name]
        
        # Handle Mega evolutions
        if 'Mega ' in name:
            base_name = name.replace('Mega ', '')
            base_form = pokemon_df[pokemon_df['Name'].str.contains(base_name.split()[0])]
            if len(base_form) > 0:
                # Find the base form (non-mega)
                for _, row in base_form.iterrows():
                    if 'Mega' not in row['Name']:
                        return row['#']
                # If no non-mega form found, use the first result
                return base_form.iloc[0]['#']
        
        # For forms like "Aegislash Blade Forme", use the base ID
        if ' Forme' in name or ' Form' in name or ' Mode' in name:
            base_name = name.split()[0]
            base_form = pokemon_df[pokemon_df['Name'].str.contains(base_name)]
            if len(base_form) > 0:
                return base_form.iloc[0]['#']
        
        # For special cases like "Kyurem Black Kyurem", use the base ID
        if len(name.split()) > 1 and name.split()[0] in name.split()[1:]:
            base_name = name.split()[0]
            base_form = pokemon_df[pokemon_df['Name'] == base_name]
            if len(base_form) > 0:
                return base_form.iloc[0]['#']
        
        # Return the original ID if no special case applies
        return pokemon_id
    except Exception as e:
        print(f"Error mapping Pokémon ID: {e}")
        return pokemon_id

def get_pokemon_sprite_url(pokemon_id):
    """Get the URL for a Pokémon sprite"""
    # Map to National Dex ID for proper sprite fetching
    national_dex_id = map_to_national_dex_id(pokemon_id)
    
    print(f"Mapping Pokémon ID {pokemon_id} to National Dex ID {national_dex_id}")
    
    # Use the PokéAPI format for sprites
    return f"https://raw.githubusercontent.com/PokeAPI/sprites/master/sprites/pokemon/{national_dex_id}.png"

def download_pokemon_sprite(pokemon_id):
    """Download a Pokémon sprite and save it locally"""
    sprite_url = get_pokemon_sprite_url(pokemon_id)
    sprite_path = os.path.join(SPRITES_DIR, f"pokemon_{pokemon_id}.png")
    
    # Check if we already have the sprite
    if os.path.exists(sprite_path):
        return sprite_path
    
    # Download the sprite
    try:
        response = requests.get(sprite_url)
        if response.status_code == 200:
            with open(sprite_path, 'wb') as f:
                f.write(response.content)
            print(f"Successfully downloaded sprite for Pokémon {pokemon_id}")
            return sprite_path
        else:
            print(f"Failed to download sprite for Pokémon {pokemon_id}: Status code {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading sprite for Pokémon {pokemon_id}: {e}")
        return None

def display_pokemon_team(team_ids, title="Pokémon Team"):
    """Display a Pokémon team with sprites"""
    pokemon_df = load_pokemon_data()
    if pokemon_df is None:
        return None
    
    fig, axes = plt.subplots(1, len(team_ids), figsize=(len(team_ids)*2, 3))
    fig.suptitle(title)
    
    if len(team_ids) == 1:
        axes = [axes]  # Make it iterable for single Pokémon
    
    for i, pokemon_id in enumerate(team_ids):
        sprite_path = download_pokemon_sprite(pokemon_id)
        
        # Get Pokémon name
        pokemon_name = "Unknown"
        pokemon_row = pokemon_df[pokemon_df['#'] == pokemon_id]
        if len(pokemon_row) > 0:
            pokemon_name = pokemon_row['Name'].values[0]
        
        if sprite_path and os.path.exists(sprite_path):
            img = Image.open(sprite_path)
            axes[i].imshow(img)
            axes[i].set_title(pokemon_name, fontsize=8)
        else:
            axes[i].text(0.5, 0.5, f"#{pokemon_id}", ha='center', va='center')
            axes[i].set_title(pokemon_name, fontsize=8)
        
        axes[i].axis('off')
    
    plt.tight_layout()
    return fig

def display_battle_result(team1_ids, team2_ids, win_probability, winner=None):
    """Display a battle result with both teams and the winner"""
    pokemon_df = load_pokemon_data()
    if pokemon_df is None:
        return None
    
    # Create figure with two rows
    fig, axes = plt.subplots(2, max(len(team1_ids), len(team2_ids)), 
                             figsize=(max(len(team1_ids), len(team2_ids))*2, 6))
    
    # Handle case with single Pokémon
    if max(len(team1_ids), len(team2_ids)) == 1:
        axes = axes.reshape(2, 1)
    
    # Team 1 (top row)
    for i in range(len(axes[0])):
        if i < len(team1_ids):
            pokemon_id = team1_ids[i]
            sprite_path = download_pokemon_sprite(pokemon_id)
            
            # Get Pokémon name
            pokemon_name = "Unknown"
            pokemon_row = pokemon_df[pokemon_df['#'] == pokemon_id]
            if len(pokemon_row) > 0:
                pokemon_name = pokemon_row['Name'].values[0]
            
            if sprite_path and os.path.exists(sprite_path):
                img = Image.open(sprite_path)
                axes[0, i].imshow(img)
                axes[0, i].set_title(pokemon_name, fontsize=8)
            else:
                axes[0, i].text(0.5, 0.5, f"#{pokemon_id}", ha='center', va='center')
                axes[0, i].set_title(pokemon_name, fontsize=8)
        
        axes[0, i].axis('off')
    
    # Team 2 (bottom row)
    for i in range(len(axes[1])):
        if i < len(team2_ids):
            pokemon_id = team2_ids[i]
            sprite_path = download_pokemon_sprite(pokemon_id)
            
            # Get Pokémon name
            pokemon_name = "Unknown"
            pokemon_row = pokemon_df[pokemon_df['#'] == pokemon_id]
            if len(pokemon_row) > 0:
                pokemon_name = pokemon_row['Name'].values[0]
            
            if sprite_path and os.path.exists(sprite_path):
                img = Image.open(sprite_path)
                axes[1, i].imshow(img)
                axes[1, i].set_title(pokemon_name, fontsize=8)
            else:
                axes[1, i].text(0.5, 0.5, f"#{pokemon_id}", ha='center', va='center')
                axes[1, i].set_title(pokemon_name, fontsize=8)
        
        axes[1, i].axis('off')
    
    # Add win probability
    team1_prob = win_probability
    team2_prob = 1 - win_probability
    
    if winner == 1:
        fig.suptitle(f"Team 1 wins! ({team1_prob:.1%} probability)", fontsize=14, color='green')
    elif winner == 2:
        fig.suptitle(f"Team 2 wins! ({team2_prob:.1%} probability)", fontsize=14, color='green')
    else:
        fig.suptitle(f"Battle Prediction: Team 1: {team1_prob:.1%}, Team 2: {team2_prob:.1%}", fontsize=12)
    
    plt.tight_layout()
    return fig 