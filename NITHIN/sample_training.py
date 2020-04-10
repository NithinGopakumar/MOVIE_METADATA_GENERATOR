from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding


# new entity label
LABEL = "DESIGNATION"

# training data
# Note: If you're using an existing model, make sure to mix in examples of
# other entity types that spaCy correctly recognized before. Otherwise, your
# model might learn the new type, but "forget" what it previously knew.
# https://explosion.ai/blog/pseudo-rehearsal-catastrophic-forgetting
TRAIN_DATA = [
    ("Unit Production Manager,SUSAN McNAMARA", {"entities": [(0, 23, 'DESIGNATION'), (24, 38, "PERSON")]}),

    ("Unit Production Manager,DAVID VALDES", {"entities": [(0, 23, 'DESIGNATION'), (24, 36, "PERSON")]}),

    ("First Assistant Director,BRIAN BETTWY", {"entities": [(0, 24, 'DESIGNATION'), (25, 37, "PERSON")]}),

    ("Second Assistant Director,DAVID VINCENT RIMER", {"entities": [(0, 25, 'DESIGNATION'), (26, 45, "LOC")]}),

    ("Post Production Supervisor,TOM PROPER", {"entities": [(0, 26, 'DESIGNATION'), (27, 37, "PERSON")]}),

    ("Supervising First Assistant Editor,JASON GAUDIO", {"entities": [(0, 34, 'DESIGNATION'), (35, 47, "LOC")]}),

    ("Production Accountant,DAWN ROBINETTE", {"entities": [(0, 21, 'DESIGNATION'), (22, 36, "PERSON")]}),

    ("Stunt Coordinator,GARRETT WARREN", {"entities": [(0, 17, 'DESIGNATION'), (18, 32, "LOC")]}),

    ("Cast", {"entities": [(0, 5, 'DESIGNATION')]}),
    ('Directed by,Adil El Arbi', {'entities': [(0, 11, 'DESIGNATION'), (12, 24, 'PERSON')]}),
    ('Directed by,Bilall Fallah', {'entities': [(0, 11, 'DESIGNATION'), (12, 25, 'PERSON')]}),
    ('Produced by,Jerry Bruckheimer', {'entities': [(0, 11, 'DESIGNATION'), (12, 29, 'PERSON')]}),
    ('Produced by,Will Smith', {'entities': [(0, 11, 'DESIGNATION'), (12, 22, 'PERSON')]}),
    ('Produced by,Doug Belgrad', {'entities': [(0, 11, 'DESIGNATION'), (12, 24, 'PERSON')]}),
    ('Screenplay by,Chris Bremner', {'entities': [(0, 13, 'DESIGNATION'), (14, 27, 'PERSON')]}),
    ('Screenplay by,Peter Craig', {'entities': [(0, 13, 'DESIGNATION'), (14, 25, 'PERSON')]}),
    ('Screenplay by,Joe Carnahan', {'entities': [(0, 13, 'DESIGNATION'), (14, 26, 'PERSON')]}),
    ('Story by,Peter Craig', {'entities': [(0, 8, 'DESIGNATION'), (9, 20, 'PERSON')]}),
    ('Story by,Joe Carnahan', {'entities': [(0, 8, 'DESIGNATION'), (9, 21, 'PERSON')]}),
    ('Based on,Characters', {'entities': [(0, 8, 'DESIGNATION'), (9, 19, 'PERSON')]}),
    ('Based on,by George Gallo', {'entities': [(0, 8, 'DESIGNATION'), (9, 24, 'PERSON')]}),
    ('Music by,Lorne Balfe', {'entities': [(0, 8, 'DESIGNATION'), (9, 20, 'PERSON')]}),
    ('Cinematography,Robrecht Heyvaert', {'entities': [(0, 14, 'DESIGNATION'), (15, 32, 'PERSON')]}),
    ('Edited by,Dan Lebental', {'entities': [(0, 9, 'DESIGNATION'), (10, 22, 'PERSON')]}),
    ('Edited by,Peter McNulty', {'entities': [(0, 9, 'DESIGNATION'), (10, 23, 'PERSON')]}),
    ('Production companies,Columbia Pictures', {'entities': [(0, 20, 'DESIGNATION'), (21, 38, 'PERSON')]}),
    ('Production companies,2.0 Entertainment', {'entities': [(0, 20, 'DESIGNATION'), (21, 38, 'PERSON')]}),
    ('Production companies,Don Simpson/Jerry', {'entities': [(0, 20, 'DESIGNATION'), (21, 38, 'PERSON')]}),
    ('Production companies,Bruckheimer Films', {'entities': [(0, 20, 'DESIGNATION'), (21, 38, 'PERSON')]}),
    ('Production companies,Overbrook Entertainment', {'entities': [(0, 20, 'DESIGNATION'), (21, 44, 'PERSON')]}),
    ('Distributed by,Sony Pictures Releasing', {'entities': [(0, 14, 'DESIGNATION'), (15, 38, 'PERSON')]}),
    ('Directed by,J. J. Abrams', {'entities': [(0, 11, 'DESIGNATION'), (12, 24, 'PERSON')]}), (
        'Produced by,Kathleen Kennedy,J. J. Abrams,Michelle Rejwan',
        {'entities': [(0, 11, 'DESIGNATION'), (12, 57, 'PERSON')]}),
    ('Screenplay by,Chris Terrio', {'entities': [(0, 13, 'DESIGNATION'), (14, 26, 'PERSON')]}),
    ('Screenplay by,J. J. Abrams', {'entities': [(0, 13, 'DESIGNATION'), (14, 26, 'PERSON')]}),
    ('Story by,Derek Connolly', {'entities': [(0, 8, 'DESIGNATION'), (9, 23, 'PERSON')]}),
    ('Story by,Colin Trevorrow', {'entities': [(0, 8, 'DESIGNATION'), (9, 24, 'PERSON')]}),
    ('Story by,J. J. Abrams,Chris Terrio', {'entities': [(0, 8, 'DESIGNATION'), (9, 34, 'PERSON')]}),
    ('Based on Characters by, George Lucas', {'entities': [(0, 22, 'DESIGNATION'), (23, 36, 'PERSON')]}),
    ('Music by,John Williams', {'entities': [(0, 8, 'DESIGNATION'), (9, 22, 'PERSON')]}),
    ('Cinematography,Dan Mindel', {'entities': [(0, 14, 'DESIGNATION'), (15, 25, 'PERSON')]}),
    ('Edited by,Maryann Brandon', {'entities': [(0, 9, 'DESIGNATION'), (10, 25, 'PERSON')]}),
    ('Edited by,Stefan Grube', {'entities': [(0, 9, 'DESIGNATION'), (10, 22, 'PERSON')]}),
    ('Production company,Lucasfilm Ltd.', {'entities': [(0, 18, 'DESIGNATION'), (19, 33, 'PERSON')]}),
    ('Production company,Bad Robot Productions', {'entities': [(0, 18, 'DESIGNATION'), (19, 40, 'PERSON')]}),
    ('Distributed by,Walt Disney Studios', {'entities': [(0, 14, 'DESIGNATION'), (15, 34, 'PERSON')]}),
    ('Distributed by,Motion Pictures', {'entities': [(0, 14, 'DESIGNATION'), (15, 30, 'PERSON')]}),
    ('MUSIC By,JOHN PAESANO', {'entities': [(0, 8, 'DESIGNATION'), (9, 21, 'PERSON')]}),
    ('UNIT PROOUCTION MANAGER,JAMES POWERS', {'entities': [(0, 23, 'DESIGNATION'), (24, 36, 'PERSON')]}),
    ('FIRST ASSISTANT DIRECTOR,VINCENT LASCOUMES', {'entities': [(0, 24, 'DESIGNATION'), (25, 42, 'PERSON')]}),
    ('FIRST ASSISTANT DIRECTOR,JEFF JJ AUTHORS', {'entities': [(0, 24, 'DESIGNATION'), (25, 40, 'PERSON')]}),
    ('SECOND ASSISTANT DIRECTORS,NADIA BRAND', {'entities': [(0, 26, 'DESIGNATION'), (27, 38, 'PERSON')]}),
    ('SECOND ASSISTANT DIRECTORS,JOY HOES', {'entities': [(0, 26, 'DESIGNATION'), (27, 35, 'PERSON')]}),
    ('SECOND ASSISTANT DIRECTORS,DAVID KLOHN', {'entities': [(0, 26, 'DESIGNATION'), (27, 38, 'PERSON')]}),
    ('STUNT COORDINATOR,GREG KRIEK', {'entities': [(0, 17, 'DESIGNATION'), (18, 28, 'PERSON')]}),
    ('ASSISTANT STUNT COORDINATOR,DAVID BUTLER', {'entities': [(0, 27, 'DESIGNATION'), (28, 40, 'PERSON')]}),
    ('FIGHT CHOREOGRAPHER,LIZA SCHOLTZ', {'entities': [(0, 19, 'DESIGNATION'), (20, 32, 'PERSON')]}),
    ('Unit Production Manager,SUSAN McNAMARA', {'entities': [(0, 23, 'DESIGNATION'), (24, 38, 'PERSON')]}),
    ('Unit Production Manager,DAVID VALDES', {'entities': [(0, 23, 'DESIGNATION'), (24, 36, 'PERSON')]}),
    ('First Assistant Director,BRIAN BETTWY', {'entities': [(0, 24, 'DESIGNATION'), (25, 37, 'PERSON')]}),
    ('Second Assistant Director,DAVID VINCENT RIMER', {'entities': [(0, 25, 'DESIGNATION'), (26, 45, 'PERSON')]}),
    ('Production Accountant,DAWN ROBINETTE', {'entities': [(0, 21, 'DESIGNATION'), (22, 36, 'PERSON')]}),
    ('Stunt Coordinator,GARRETT WARREN', {'entities': [(0, 17, 'DESIGNATION'), (18, 32, 'PERSON')]}),
    ('WETA Animation Supervisor,MICHAEL COZENS', {'entities': [(0, 25, 'DESIGNATION'), (26, 40, 'PERSON')]}),
    ('Co-Production Designer,CAYLAH EDDLEBLUTE', {'entities': [(0, 22, 'DESIGNATION'), (23, 40, 'PERSON')]}),
    ('Supervising Art Director,A.TODD HOLLAND', {'entities': [(0, 24, 'DESIGNATION'), (25, 39, 'PERSON')]}),
    ('Post Production Supervisor,TOM PROPER', {'entities': [(0, 26, 'DESIGNATION'), (27, 37, 'PERSON')]}),
    ('Supervising First Assistant Editor,JASON GAUDIO', {'entities': [(0, 34, 'DESIGNATION'), (35, 47, 'PERSON')]}),
    ('Assistant Stunt Coordinator,STEVE BROWN', {'entities': [(0, 27, 'DESIGNATION'), (28, 39, 'PERSON')]}),
    ('Fight Choreographer,STEVE BROWN', {'entities': [(0, 19, 'DESIGNATION'), (20, 31, 'PERSON')]}),
    ('Art Directors,LESLIE McDONALD', {'entities': [(0, 13, 'DESIGNATION'), (14, 29, 'PERSON')]}),
    ('Assistant Art Director,PAUL ALIX', {'entities': [(0, 22, 'DESIGNATION'), (23, 32, 'PERSON')]}),
    ('Storyboard Artists,JIM MITCHELL', {'entities': [(0, 18, 'DESIGNATION'), (19, 31, 'PERSON')]}),
    ('Set Designers,JOHN BERGER', {'entities': [(0, 13, 'DESIGNATION'), (14, 25, 'PERSON')]}),
    ('Production Supervisor,NATALIE ANGEL', {'entities': [(0, 21, 'DESIGNATION'), (22, 35, 'PERSON')]}),
    ('Production Coordinator,CYNTHIA STREIT', {'entities': [(0, 22, 'DESIGNATION'), (23, 37, 'PERSON')]}),
    ('Adil El Arbi,Directed by', {'entities': [(0, 12, 'PERSON'), (13, 24, 'DESIGNATION')]}),
    ('Bilall Fallah,Directed by', {'entities': [(0, 13, 'PERSON'), (14, 25, 'DESIGNATION')]}),
    ('Jerry Bruckheimer,Produced by', {'entities': [(0, 17, 'PERSON'), (18, 29, 'DESIGNATION')]}),
    ('Will Smith,Produced by', {'entities': [(0, 10, 'PERSON'), (11, 22, 'DESIGNATION')]}),
    ('Doug Belgrad,Produced by', {'entities': [(0, 12, 'PERSON'), (13, 24, 'DESIGNATION')]}),
    ('Chris Bremner,Screenplay by', {'entities': [(0, 13, 'PERSON'), (14, 27, 'DESIGNATION')]}),
    ('Peter Craig,Screenplay by', {'entities': [(0, 11, 'PERSON'), (12, 25, 'DESIGNATION')]}),
    ('Joe Carnahan,Screenplay by', {'entities': [(0, 12, 'PERSON'), (13, 26, 'DESIGNATION')]}),
    ('Peter Craig,Story by', {'entities': [(0, 11, 'PERSON'), (12, 20, 'DESIGNATION')]}),
    ('Joe Carnahan,Story by', {'entities': [(0, 12, 'PERSON'), (13, 21, 'DESIGNATION')]}),
    ('Characters,Based on', {'entities': [(0, 10, 'PERSON'), (11, 19, 'DESIGNATION')]}),
    ('George Gallo,Based on by', {'entities': [(0, 13, 'PERSON'), (14, 25, 'DESIGNATION')]}),
    ('Lorne Balfe,Music by', {'entities': [(0, 11, 'PERSON'), (12, 20, 'DESIGNATION')]}),
    ('Robrecht Heyvaert,Cinematography', {'entities': [(0, 17, 'PERSON'), (18, 32, 'DESIGNATION')]}),
    ('Dan Lebental,Edited by', {'entities': [(0, 12, 'PERSON'), (13, 22, 'DESIGNATION')]}),
    ('Peter McNulty,Edited by', {'entities': [(0, 13, 'PERSON'), (14, 23, 'DESIGNATION')]}),
    ('Columbia Pictures,Production companies', {'entities': [(0, 17, 'PERSON'), (18, 38, 'DESIGNATION')]}),
    ('2.0 Entertainment,Production companies', {'entities': [(0, 17, 'PERSON'), (18, 38, 'DESIGNATION')]}),
    ('Don Simpson/Jerry,Production companies', {'entities': [(0, 17, 'PERSON'), (18, 38, 'DESIGNATION')]}),
    ('Bruckheimer Films,Production companies', {'entities': [(0, 17, 'PERSON'), (18, 38, 'DESIGNATION')]}),
    ('Overbrook Entertainment,Production companies', {'entities': [(0, 23, 'PERSON'), (24, 44, 'DESIGNATION')]}),
    ('Sony Pictures Releasing,Distributed by', {'entities': [(0, 23, 'PERSON'), (24, 38, 'DESIGNATION')]}),
    ('J. J. Abrams,Directed by', {'entities': [(0, 12, 'PERSON'), (13, 24, 'DESIGNATION')]}), (
        'Kathleen Kennedy,J. J. Abrams,Michelle Rejwan,Produced by',
        {'entities': [(0, 16, 'PERSON'), (17, 57, 'DESIGNATION')]}),
    ('Chris Terrio,Screenplay by', {'entities': [(0, 12, 'PERSON'), (13, 26, 'DESIGNATION')]}),
    ('J. J. Abrams,Screenplay by', {'entities': [(0, 12, 'PERSON'), (13, 26, 'DESIGNATION')]}),
    ('Derek Connolly,Story by', {'entities': [(0, 14, 'PERSON'), (15, 23, 'DESIGNATION')]}),
    ('Colin Trevorrow,Story by', {'entities': [(0, 15, 'PERSON'), (16, 24, 'DESIGNATION')]}),
    ('J. J. Abrams,Chris Terrio,Story by', {'entities': [(0, 12, 'PERSON'), (13, 34, 'DESIGNATION')]}),
    ('George Lucas,Based on Characters by', {'entities': [(0, 13, 'PERSON'), (14, 36, 'DESIGNATION')]}),
    ('John Williams,Music by', {'entities': [(0, 13, 'PERSON'), (14, 22, 'DESIGNATION')]}),
    ('Dan Mindel,Cinematography', {'entities': [(0, 10, 'PERSON'), (11, 25, 'DESIGNATION')]}),
    ('Maryann Brandon,Edited by', {'entities': [(0, 15, 'PERSON'), (16, 25, 'DESIGNATION')]}),
    ('Stefan Grube,Edited by', {'entities': [(0, 12, 'PERSON'), (13, 22, 'DESIGNATION')]}),
    ('Lucasfilm Ltd.,Production company', {'entities': [(0, 14, 'PERSON'), (15, 33, 'DESIGNATION')]}),
    ('Bad Robot Productions,Production company', {'entities': [(0, 21, 'PERSON'), (22, 40, 'DESIGNATION')]}),
    ('Walt Disney Studios,Distributed by', {'entities': [(0, 19, 'PERSON'), (20, 34, 'DESIGNATION')]}),
    ('Motion Pictures,Distributed by', {'entities': [(0, 15, 'PERSON'), (16, 30, 'DESIGNATION')]}),
    ('JOHN PAESANO,MUSIC By', {'entities': [(0, 12, 'PERSON'), (13, 21, 'DESIGNATION')]}),
    ('JAMES POWERS,UNIT PROOUCTION MANAGER', {'entities': [(0, 12, 'PERSON'), (13, 36, 'DESIGNATION')]}),
    ('VINCENT LASCOUMES,FIRST ASSISTANT DIRECTOR', {'entities': [(0, 17, 'PERSON'), (18, 42, 'DESIGNATION')]}),
    ('JEFF JJ AUTHORS,FIRST ASSISTANT DIRECTOR', {'entities': [(0, 15, 'PERSON'), (16, 40, 'DESIGNATION')]}),
    ('NADIA BRAND,SECOND ASSISTANT DIRECTORS', {'entities': [(0, 11, 'PERSON'), (12, 38, 'DESIGNATION')]}),
    ('JOY HOES,SECOND ASSISTANT DIRECTORS', {'entities': [(0, 8, 'PERSON'), (9, 35, 'DESIGNATION')]}),
    ('DAVID KLOHN,SECOND ASSISTANT DIRECTORS', {'entities': [(0, 11, 'PERSON'), (12, 38, 'DESIGNATION')]}),
    ('GREG KRIEK,STUNT COORDINATOR', {'entities': [(0, 10, 'PERSON'), (11, 28, 'DESIGNATION')]}),
    ('DAVID BUTLER,ASSISTANT STUNT COORDINATOR', {'entities': [(0, 12, 'PERSON'), (13, 40, 'DESIGNATION')]}),
    ('LIZA SCHOLTZ,FIGHT CHOREOGRAPHER', {'entities': [(0, 12, 'PERSON'), (13, 32, 'DESIGNATION')]}),
    ('SUSAN McNAMARA,Unit Production Manager', {'entities': [(0, 14, 'PERSON'), (15, 38, 'DESIGNATION')]}),
    ('DAVID VALDES,Unit Production Manager', {'entities': [(0, 12, 'PERSON'), (13, 36, 'DESIGNATION')]}),
    ('BRIAN BETTWY,First Assistant Director', {'entities': [(0, 12, 'PERSON'), (13, 37, 'DESIGNATION')]}),
    ('DAVID VINCENT RIMER,Second Assistant Director', {'entities': [(0, 19, 'PERSON'), (20, 45, 'DESIGNATION')]}),
    ('DAWN ROBINETTE,Production Accountant', {'entities': [(0, 14, 'PERSON'), (15, 36, 'DESIGNATION')]}),
    ('GARRETT WARREN,Stunt Coordinator', {'entities': [(0, 14, 'PERSON'), (15, 32, 'DESIGNATION')]}),
    ('MICHAEL COZENS,WETA Animation Supervisor', {'entities': [(0, 14, 'PERSON'), (15, 40, 'DESIGNATION')]}),
    ('CAYLAH EDDLEBLUTE,Co-Production Designer', {'entities': [(0, 17, 'PERSON'), (18, 40, 'DESIGNATION')]}),
    ('A.TODD HOLLAND,Supervising Art Director', {'entities': [(0, 14, 'PERSON'), (15, 39, 'DESIGNATION')]}),
    ('TOM PROPER,Post Production Supervisor', {'entities': [(0, 10, 'PERSON'), (11, 37, 'DESIGNATION')]}),
    ('JASON GAUDIO,Supervising First Assistant Editor', {'entities': [(0, 12, 'PERSON'), (13, 47, 'DESIGNATION')]}),
    ('STEVE BROWN,Assistant Stunt Coordinator', {'entities': [(0, 11, 'PERSON'), (12, 39, 'DESIGNATION')]}),
    ('STEVE BROWN,Fight Choreographer', {'entities': [(0, 11, 'PERSON'), (12, 31, 'DESIGNATION')]}),
    ('LESLIE McDONALD,Art Directors', {'entities': [(0, 15, 'PERSON'), (16, 29, 'DESIGNATION')]}),
    ('PAUL ALIX,Assistant Art Director', {'entities': [(0, 9, 'PERSON'), (10, 32, 'DESIGNATION')]}),
    ('JIM MITCHELL,Storyboard Artists', {'entities': [(0, 12, 'PERSON'), (13, 31, 'DESIGNATION')]}),
    ('JOHN BERGER,Set Designers', {'entities': [(0, 11, 'PERSON'), (12, 25, 'DESIGNATION')]}),
    ('NATALIE ANGEL,Production Supervisor', {'entities': [(0, 13, 'PERSON'), (14, 35, 'DESIGNATION')]}),
    ('CYNTHIA STREIT,Production Coordinator', {'entities': [(0, 14, 'PERSON'), (15, 37, 'DESIGNATION')]}),
    ("Directed by,Robert Rodriguez", {"entities": [(0, 11, "DESIGNATION"), (12, 28, "PERSON")]}),

    ("Produced by,James Cameron,Jon Landau",
     {"entities": [(0, 11, "DESIGNATION"), (12, 25, "PERSON"), (26, 36, "PERSON")]}),

    ("Screenplay by,James Cameron,Laeta Kalogridis",
     {"entities": [(0, 13, "DESIGNATION"), (14, 27, "PERSON"), (28, 44, "PERSON")]}),

    ("Edited bY,Stephen E. Rivkin,Ian Silverstein",
     {"entities": [(0, 9, "DESIGNATION"), (10, 27, "PERSON"), (28, 43, "PERSON")]}),

    ("Associate Producers,RICK PORRAS,STEVEN J. BOYD,ROB LOWE,JAKE BUSEY", {"entities": [(0, 19, "DESIGNATION"),
                                                                                         (20, 31, "PERSON"),
                                                                                         (32, 46, "PERSON"),
                                                                                         (47, 55, "PERSON"),
                                                                                         (56, 66, "PERSON")]}),
    ("Screenplay by,JAMES V. HART and MICHAEL GOLDENBERG",
     {"entities": [(0, 13, "DESIGNATION"), (14, 27, "PERSON"), (32, 50, "PERSON")]}),

    ("A ROBERT ZEMECKIS Film", {"entities": [(2, 17, "PERSON")]}),
    ("Based on the Novel by,CARL SAGAN", {"entities": [(0, 21, "DESIGNATION"), (22, 32, "PERSON")]}),

    ("Director of Photography,DON BURGESS", {"entities": [(0, 23, "DESIGNATION"), (24, 35, "PERSON")]}),

    ("Production Designer,ED VERREAUX", {"entities": [(0, 19, "DESIGNATION"), (20, 31, "PERSON")]}),

    ("Edited by,ARTHUR SCHMIDT", {"entities": [(0, 9, "DESIGNATION"), (10, 24, "PERSON")]}),

    ("Music by,ALAN SILVESTRI", {"entities": [(0, 8, "DESIGNATION"), (9, 23, "PERSON")]}),

    ("Directed by,ROBERT ZEMECKIS", {"entities": [(0, 11, "DESIGNATION"), (12, 27, "PERSON")]}),

    ("Senior Visual Effects Supervisor,KEN RALSTON", {"entities": [(0, 32, "DESIGNATION"), (33, 44, "PERSON")]}),

    ("Second Assistant Director,CELLIN GLUCK", {"entities": [(0, 25, "DESIGNATION"), (26, 38, "PERSON")]}),

    ("Second Second Assistant Director,DARIN RIVETTI", {"entities": [(0, 32, "DESIGNATION"), (33, 46, "PERSON")]}),

    ("Costumes Designed by,JOANNA JOHNSTON", {"entities": [(0, 20, "DESIGNATION"), (21, 36, "PERSON")]}),

    ("Casting by,VICTORIA BURROWS", {"entities": [(0, 10, "DESIGNATION"), (11, 27, "PERSON")]}),

    ("Co-Producers,CARL SAGAN and ANN DRUYAN", {"entities": [(0, 12, "DESIGNATION"), (13, 38, "PERSON")]}),
    ("Unit Production Manager,JOAN BRADSHAW", {"entities": [(0, 23, "DESIGNATION"), (24, 37, "PERSON")]}),

    ("First Assistant Director,BRUCE MORIARTY", {"entities": [(0, 24, "DESIGNATION"), (25, 39, "PERSON")]}),

    ("Unit Production Manager,CHERYLANNE MARTIN", {"entities": [(0, 23, "DESIGNATION"), (24, 41, "PERSON")]}),

    ("Art Directors,LAWRENCE A. HUBBS,BRUCE CRONE",
     {"entities": [(0, 13, "DESIGNATION"), (14, 31, "PERSON"), (32, 43, "PERSON")]}),

    ("Video Graphics Supervisor,IAN KELLY", {"entities": [(0, 25, "DESIGNATION"), (26, 35, "PERSON")]}),
    ("Assistan Editors,FRED VITALE,JANA GOLD,MATHEW SCHMIDT,SEAN MENZIES", {
        "entities": [(0, 16, "DESIGNATION"), (17, 28, "PERSON"), (29, 38, "PERSON"), (39, 53, "PERSON"),
                     (54, 66, "PERSON")]}),
    ("Directed by,Adil El Arbi,Bilall Fallah",
     {"entities": [(0, 11, "DESIGNATION"), (12, 24, "PERSON"), (25, 38, "PERSON")]}),
    ("Produced by,Jerry Bruckheimer,Will Smith,Doug Belgrad", {"entities": [(0, 11, "DESIGNATION"),
                                                                            (12, 29, "PERSON"), (30, 40, "PERSON"),
                                                                            (41, 53, "PERSON")]}),
    ("STUNT RIGGERS,ULI RITCHER,NIKLAS KINZEL", {"entities": [(0, 13, "DESIGNATION"),
                                                              (14, 25, "PERSON"),
                                                              (26, 39, "PERSON")]}),

    (
        "CO-PRODUCERS CHRISTOPH FISSER,HENNING MOLFENTER,CHARLIE WOEBCKEN",
        {"entities": [(0, 12, "DESIGNATION"), (13, 29, "PERSON"),
                      (30, 47, "PERSON"), (48, 64, "PERSON")]}),

    ("Associate Producers,ARI HANDEL,EVAN GINZBURG",
     {"entities": [(0, 19, "DESIGNATION"), (20, 30, "PERSON"), (31, 44, "PERSON")]}),
    ("Sound Supervision and Design,JACOB RIBICOFF,BRIAN EMRICH",
     {"entities": [(0, 28, "DESIGNATION"), (29, 43, "PERSON"), (44, 56, "PERSON")]}),

    ("First Assistant Camera,JEFF DUTEMPLE,MALCOM PURNELL",
     {"entities": [(0, 22, "DESIGNATION"), (23, 36, "PERSON"), (37, 51, "PERSON")]}),

    ("Second Assistant Camera,DANIEL WIENER,TRAVIS CADALZO",
     {"entities": [(0, 23, "DESIGNATION"), (24, 37, "PERSON"), (38, 52, "PERSON")]}),

    ("Genny Operators,ROBERT GURGO,WILLIAM HINES",
     {"entities": [(0, 15, "DESIGNATION"), (16, 28, "PERSON"), (29, 42, "PERSON")]}),

    ("Rigging Gaffers,MICHAEL GALLART,TIMOTHY HEALY",
     {"entities": [(0, 15, "DESIGNATION"), (16, 31, "PERSON"), (32, 45, "PERSON")]}),

    ('Unit Production Manager,SUSAN McNAMARA', {'entities': [(0, 23, 'DESIGNATION'), (24, 38, 'PERSON')]}),
    ('Unit Production Manager,DAVID VALDES', {'entities': [(0, 23, 'DESIGNATION'), (24, 36, 'PERSON')]}),
    ('First Assistant Director,BRIAN BETTWY', {'entities': [(0, 24, 'DESIGNATION'), (25, 37, 'PERSON')]}),
    ('Unit Production Manager,Cristen Carr Strubbe', {'entities': [(0, 23, 'DESIGNATION'), (24, 44, 'PERSON')]}),
    ('First Assistant Director,Sergio Mimica-Gezzan', {'entities': [(0, 24, 'DESIGNATION'), (25, 45, 'PERSON')]}),
    ('Second Assistant Director,David H. Venghaus Jr.', {'entities': [(0, 25, 'DESIGNATION'), (26, 47, 'PERSON')]}),
    ('DRIVING INSTRUCTOR,DANICA PATRICK', {'entities': [(0, 18, 'DESIGNATION'), (19, 33, 'PERSON')]}),
    ('FIGHT INSTRUCTOR,RONDA ROUSEY', {'entities': [(0, 16, 'DESIGNATION'), (17, 29, 'PERSON')]}),
    ('BOMB INSTRUCTOR,LAVERENE COX', {'entities': [(0, 15, 'DESIGNATION'), (16, 28, 'PERSON')]}),
    ('STUNT COORDINATOR,FLORIAN HOTZ', {'entities': [(0, 17, 'DESIGNATION'), (18, 30, 'PERSON')]}),
    ('ASSISTANT STUNT COORDINATOR,SANDRA BARGER', {'entities': [(0, 27, 'DESIGNATION'), (28, 41, 'PERSON')]}),
    ('FIGHT CHOREOGRAPHER,TOLGA DEGIRMEN', {'entities': [(0, 19, 'DESIGNATION'), (20, 34, 'PERSON')]}),
    ('HEAD STUNT RIGGER,ALEXANDER MAGERL', {'entities': [(0, 17, 'DESIGNATION'), (18, 34, 'PERSON')]}),
    ('UNIT PRODUCTION MANAGER,ARNO NEUBAUER', {'entities': [(0, 23, 'DESIGNATION'), (24, 37, 'PERSON')]}),
    ('FIRST ASSISTANT DIRECTOR,ALEX OAKLEY', {'entities': [(0, 24, 'DESIGNATION'), (25, 36, 'PERSON')]}),
    ('ASSOCIATE PRODUCER,ALEX OAKLEY', {'entities': [(0, 18, 'DESIGNATION'), (19, 30, 'PERSON')]}),
    ('SECOND ASSISTANT DIRECTOR,JAMES MANNING', {'entities': [(0, 25, 'DESIGNATION'), (26, 39, 'PERSON')]}),
    ('VISUAL EFFECTS SUPERVISOR,KAREN HESTON', {'entities': [(0, 25, 'DESIGNATION'), (26, 38, 'PERSON')]}),
    ('Unit Production Manager,JENNIFER ROTH', {'entities': [(0, 23, 'DESIGNATION'), (24, 37, 'PERSON')]}),
    ('First Assistant Director,RICHARD GRAVES', {'entities': [(0, 24, 'DESIGNATION'), (25, 39, 'PERSON')]}),
    ('Second Assistant Director,BRENDAN WALSH', {'entities': [(0, 25, 'DESIGNATION'), (26, 39, 'PERSON')]}),
    ('Stunt Coordinator,DOUGLAS CROSBY', {'entities': [(0, 17, 'DESIGNATION'), (18, 32, 'PERSON')]}),
    ('Production Supervisor,ALEXIS ARNOLD', {'entities': [(0, 21, 'DESIGNATION'), (22, 35, 'PERSON')]}),
    ('Script Supervisor,ANTHONY PETTINE', {'entities': [(0, 17, 'DESIGNATION'), (18, 33, 'PERSON')]}),
    ('Post Production Supervisor,COLLEEN BACHMAN', {'entities': [(0, 26, 'DESIGNATION'), (27, 42, 'PERSON')]}),
    ('Camera Operator,PETER NOLAN', {'entities': [(0, 15, 'DESIGNATION'), (16, 27, 'PERSON')]}),
    ('Stills Photographer,NIKO TAVERNISE', {'entities': [(0, 19, 'DESIGNATION'), (20, 34, 'PERSON')]}),
    ('24 Frame Playback,MIKE SIME', {'entities': [(0, 17, 'DESIGNATION'), (18, 27, 'PERSON')]}),
    ('Camera Intern,AURORE GUYOT', {'entities': [(0, 13, 'DESIGNATION'), (14, 26, 'PERSON')]}),
    ('Gaffer,DAVID SKUTCH', {'entities': [(0, 6, 'DESIGNATION'), (7, 19, 'PERSON')]}),
    ('Best Boy, JARAD MOLKENTHIN', {'entities': [(0, 8, 'DESIGNATION'), (9, 26, 'PERSON')]}),
    ('Rigging Electric,JAMES TEMME', {'entities': [(0, 16, 'DESIGNATION'), (17, 28, 'PERSON')]}),
    ('Theatrical Lighting Technician,MICHEAL GALLART', {'entities': [(0, 30, 'DESIGNATION'), (31, 46, 'PERSON')]}),
    ("Assistant Hair Stylists,JEFFREY REBELO,DIANA SIKES",
     {"entities": [(0, 23, "DESIGNATION"), (24, 38, "PERSON"), (39, 50, "PERSON")]}),
    ("Assistant Location Managers,NATE BRAEUER,LAUREN FRITZ",
     {"entities": [(0, 27, "DESIGNATION"), (28, 40, "PERSON"), (41, 53, "PERSON")]}),
    ("Re-Recording Mixers,DOMINICK TAVELLA,JACOB RIBICOFF",
     {"entities": [(0, 19, "DESIGNATION"), (20, 36, "PERSON"), (37, 51, "PERSON")]}),
    ("ELIZA PALEY,WILLIAM SWEENEY,Sound Effects Editors",
     {"entities": [(0, 11, "PERSON"), (12, 27, "PERSON"), (28, 49, "DESIGNATION")]}),
    ('CHRIS SKUTCH,Key Grip', {'entities': [(0, 12, 'PERSON'), (13, 21, 'DESIGNATION')]}),
    ('JOHN HALIGAN,Best Boy', {'entities': [(0, 12, 'PERSON'), (13, 21, 'DESIGNATION')]}),
    ("BEN D'ANDREA,Dolly Grip", {'entities': [(0, 12, 'PERSON'), (13, 23, 'DESIGNATION')]}),
    ('RUARK BEHAN,Grip', {'entities': [(0, 11, 'PERSON'), (12, 16, 'DESIGNATION')]}),
    ('GRAHAM KLATT,Key Rigging Grip', {'entities': [(0, 12, 'PERSON'), (13, 29, 'DESIGNATION')]}),
    ('REID KELLY,Boy Rigging Grip', {'entities': [(0, 10, 'PERSON'), (11, 27, 'DESIGNATION')]}),
    ('THEO SENA,Set Decorator', {'entities': [(0, 9, 'PERSON'), (10, 23, 'DESIGNATION')]}),
    ('SCOTT GAGNON,Leadman', {'entities': [(0, 12, 'PERSON'), (13, 20, 'DESIGNATION')]}),
    ('TIM ROSSITER,Additional Leadman', {'entities': [(0, 12, 'PERSON'), (13, 31, 'DESIGNATION')]}),
    ('JONATHAN UNGER,Set Dresser', {'entities': [(0, 14, 'PERSON'), (15, 26, 'DESIGNATION')]}),
    ('ROBIN KOENIG,On-Set Dresser', {'entities': [(0, 12, 'PERSON'), (13, 27, 'DESIGNATION')]}),
    ('TRAVIS CHILD,Sceanic', {'entities': [(0, 12, 'PERSON'), (13, 20, 'DESIGNATION')]}),
    ('JEFF BUTCHER,Prop Master', {'entities': [(0, 12, 'PERSON'), (13, 24, 'DESIGNATION')]}),
    ('DANIEL FISHER,Prop Master', {'entities': [(0, 13, 'PERSON'), (14, 25, 'DESIGNATION')]}),
    ('EOIN LAMBE,3rd Prop Assistant', {'entities': [(0, 10, 'PERSON'), (11, 29, 'DESIGNATION')]}),
    ('MATTHEW MUNN,Art Director', {'entities': [(0, 12, 'PERSON'), (13, 25, 'DESIGNATION')]}),
    ('SELINA VAN DEN BRINK,Art Department Coordinator', {'entities': [(0, 20, 'PERSON'), (21, 47, 'DESIGNATION')]}),
    ('SHANE INGERSOLL,Art Production Assistant', {'entities': [(0, 15, 'PERSON'), (16, 40, 'DESIGNATION')]}),
    ('CRAIG HENCH,Art Department intern', {'entities': [(0, 11, 'PERSON'), (12, 33, 'DESIGNATION')]}),
    ('LENORE PEMBERTON,Assistant Costume Designer', {'entities': [(0, 16, 'PERSON'), (17, 43, 'DESIGNATION')]}),
    ('STEFFANY BERNSTEIN,Wardrobe Supervisor', {'entities': [(0, 18, 'PERSON'), (19, 38, 'DESIGNATION')]}),
    ('NICCI SCHINMAN,Set Costumer', {'entities': [(0, 14, 'PERSON'), (15, 27, 'DESIGNATION')]}),
    ('CARA CZEKANSKI,Additional Costumer', {'entities': [(0, 14, 'PERSON'), (15, 34, 'DESIGNATION')]}),
    ('ESTELLA SIMMONS,Alterations', {'entities': [(0, 15, 'PERSON'), (16, 27, 'DESIGNATION')]}),
    ('RENA BUSSINGER,Costume Intern', {'entities': [(0, 14, 'PERSON'), (15, 29, 'DESIGNATION')]}),
    ('MONTE CRISTO,Ram Belt Buckle Design', {'entities': [(0, 12, 'PERSON'), (13, 35, 'DESIGNATION')]}),
    ('JUDY CHIN,Key Makeup', {'entities': [(0, 9, 'PERSON'), (10, 20, 'DESIGNATION')]}),
    ('EVE MORROW,Makeup Artist', {'entities': [(0, 10, 'PERSON'), (11, 24, 'DESIGNATION')]}),
    ('MARGIE DURAND,Assistant Makeup Artist', {'entities': [(0, 13, 'PERSON'), (14, 37, 'DESIGNATION')]}),
    ('MANDY LYONS,Key Hair Stylist', {'entities': [(0, 11, 'PERSON'), (12, 28, 'DESIGNATION')]}),
    ('MIKE MARINO,Prosthetic Makeup Designer', {'entities': [(0, 11, 'PERSON'), (12, 38, 'DESIGNATION')]}),
    ('HAYES VILANDRY,Lab Supervisor', {'entities': [(0, 14, 'PERSON'), (15, 29, 'DESIGNATION')]}),
    ('DREW JIRITANO,Special Effects', {'entities': [(0, 13, 'PERSON'), (14, 29, 'DESIGNATION')]}),
    ('KEN ISHII,Sound Mixer', {'entities': [(0, 9, 'PERSON'), (10, 21, 'DESIGNATION')]}),
    ('ANGUIBE GUINDO,Boom Operator', {'entities': [(0, 14, 'PERSON'), (15, 28, 'DESIGNATION')]}),
    ('JOE ORIGLIERI,Additional Sound B-Unit', {'entities': [(0, 13, 'PERSON'), (14, 37, 'DESIGNATION')]}),
    ('JOE ORIGLIERI,Sound Utility', {'entities': [(0, 13, 'PERSON'), (14, 27, 'DESIGNATION')]}),
    ('RYAN SMITH,Location Manager', {'entities': [(0, 10, 'PERSON'), (11, 27, 'DESIGNATION')]}),
    ('ABI JACKSON,Key Location Assistant', {'entities': [(0, 11, 'PERSON'), (12, 34, 'DESIGNATION')]}),
    ('DAN POLLACK,Location Scout', {'entities': [(0, 11, 'PERSON'), (12, 26, 'DESIGNATION')]}),
    ('ZACH THRUN,Unit PA', {'entities': [(0, 10, 'PERSON'), (11, 18, 'DESIGNATION')]}),
    ('HENRY WINNIK,Locations Interns', {'entities': [(0, 12, 'PERSON'), (13, 30, 'DESIGNATION')]}),
    ('KENNETH WACHTEL,First Assistant Editor', {'entities': [(0, 15, 'PERSON'), (16, 38, 'DESIGNATION')]}),
    ('SEBASTIAN ISCHER,Additional Assistant Editor', {'entities': [(0, 16, 'PERSON'), (17, 44, 'DESIGNATION')]}),
    ('BARRY BLASCHKE,Post Production PA', {'entities': [(0, 14, 'PERSON'), (15, 33, 'DESIGNATION')]}),
    ('SOUND ONE,Post Production Sound Services', {'entities': [(0, 9, 'PERSON'), (10, 40, 'DESIGNATION')]}),
    ('TONY MARTINEZ,ADR Supervisor', {'entities': [(0, 13, 'PERSON'), (14, 28, 'DESIGNATION')]}),
    ('TONY MARTINEZ,Dialoague Supervisor', {'entities': [(0, 13, 'PERSON'), (14, 34, 'DESIGNATION')]}),
    ('DANIEL KORINTUS,Dialogue Editor', {'entities': [(0, 15, 'PERSON'), (16, 31, 'DESIGNATION')]}),
    ('STUART STANLEY,Foley Editor', {'entities': [(0, 14, 'PERSON'), (15, 27, 'DESIGNATION')]}),
    ('ERIC STRAUSSER,Assistant Sound Editor', {'entities': [(0, 14, 'PERSON'), (15, 37, 'DESIGNATION')]}),
    ('DROR GESCHEIT,Machine Room Operator', {'entities': [(0, 13, 'PERSON'), (14, 35, 'DESIGNATION')]}),
    ('DAVID BOULTON,ADR Mixer', {'entities': [(0, 13, 'PERSON'), (14, 23, 'DESIGNATION')]}),
    ('MIKE HOWELLS,ADR Machine Room Operator', {'entities': [(0, 12, 'PERSON'), (13, 38, 'DESIGNATION')]}),
    ('RYAN COLLISON,Foley Mixer', {'entities': [(0, 13, 'PERSON'), (14, 25, 'DESIGNATION')]}),
    ('JAY PECK,Foley Walker', {'entities': [(0, 8, 'PERSON'), (9, 21, 'DESIGNATION')]}),
    ('ERIC VIERHAUS,Dolby Sound Consultant', {'entities': [(0, 13, 'PERSON'), (14, 36, 'DESIGNATION')]}),
    ('DAN EVANS FARKAS,Music Editor', {'entities': [(0, 16, 'PERSON'), (17, 29, 'DESIGNATION')]}),
    ('unit production manager,susan mcnamara', {'entities': [(0, 23, 'DESIGNATION'), (24, 38, 'PERSON')]}),
    ('unit production manager,david valdes', {'entities': [(0, 23, 'DESIGNATION'), (24, 36, 'PERSON')]}),
    ('first assistant director,brian bettwy', {'entities': [(0, 24, 'DESIGNATION'), (25, 37, 'PERSON')]}),
    ('second assistant director,david vincent rimer', {'entities': [(0, 25, 'DESIGNATION'), (26, 45, 'LOC')]}),
    ('post production supervisor,tom proper', {'entities': [(0, 26, 'DESIGNATION'), (27, 37, 'PERSON')]}),
    ('supervising first assistant editor,jason gaudio', {'entities': [(0, 34, 'DESIGNATION'), (35, 47, 'LOC')]}),
    ('production accountant,dawn robinette', {'entities': [(0, 21, 'DESIGNATION'), (22, 36, 'PERSON')]}),
    ('stunt coordinator,garrett warren', {'entities': [(0, 17, 'DESIGNATION'), (18, 32, 'LOC')]}),
    ('cast', {'entities': [(0, 5, 'DESIGNATION')]}),
    ('directed by,adil el arbi', {'entities': [(0, 11, 'DESIGNATION'), (12, 24, 'PERSON')]}),
    ('directed by,bilall fallah', {'entities': [(0, 11, 'DESIGNATION'), (12, 25, 'PERSON')]}),
    ('produced by,jerry bruckheimer', {'entities': [(0, 11, 'DESIGNATION'), (12, 29, 'PERSON')]}),
    ('produced by,will smith', {'entities': [(0, 11, 'DESIGNATION'), (12, 22, 'PERSON')]}),
    ('produced by,doug belgrad', {'entities': [(0, 11, 'DESIGNATION'), (12, 24, 'PERSON')]}),
    ('screenplay by,chris bremner', {'entities': [(0, 13, 'DESIGNATION'), (14, 27, 'PERSON')]}),
    ('screenplay by,peter craig', {'entities': [(0, 13, 'DESIGNATION'), (14, 25, 'PERSON')]}),
    ('screenplay by,joe carnahan', {'entities': [(0, 13, 'DESIGNATION'), (14, 26, 'PERSON')]}),
    ('story by,peter craig', {'entities': [(0, 8, 'DESIGNATION'), (9, 20, 'PERSON')]}),
    ('story by,joe carnahan', {'entities': [(0, 8, 'DESIGNATION'), (9, 21, 'PERSON')]}),
    ('based on,characters', {'entities': [(0, 8, 'DESIGNATION'), (9, 19, 'PERSON')]}),
    ('based on,by george gallo', {'entities': [(0, 8, 'DESIGNATION'), (9, 24, 'PERSON')]}),
    ('music by,lorne balfe', {'entities': [(0, 8, 'DESIGNATION'), (9, 20, 'PERSON')]}),
    ('cinematography,robrecht heyvaert', {'entities': [(0, 14, 'DESIGNATION'), (15, 32, 'PERSON')]}),
    ('edited by,dan lebental', {'entities': [(0, 9, 'DESIGNATION'), (10, 22, 'PERSON')]}),
    ('edited by,peter mcnulty', {'entities': [(0, 9, 'DESIGNATION'), (10, 23, 'PERSON')]}),
    ('production companies,columbia pictures', {'entities': [(0, 20, 'DESIGNATION'), (21, 38, 'PERSON')]}),
    ('production companies,2.0 entertainment', {'entities': [(0, 20, 'DESIGNATION'), (21, 38, 'PERSON')]}),
    ('production companies,don simpson/jerry', {'entities': [(0, 20, 'DESIGNATION'), (21, 38, 'PERSON')]}),
    ('production companies,bruckheimer films', {'entities': [(0, 20, 'DESIGNATION'), (21, 38, 'PERSON')]}),
    ('production companies,overbrook entertainment', {'entities': [(0, 20, 'DESIGNATION'), (21, 44, 'PERSON')]}),
    ('distributed by,sony pictures releasing', {'entities': [(0, 14, 'DESIGNATION'), (15, 38, 'PERSON')]}),
    ('directed by,j. j. abrams', {'entities': [(0, 11, 'DESIGNATION'), (12, 24, 'PERSON')]}), (
    'produced by,kathleen kennedy,j. j. abrams,michelle rejwan',
    {'entities': [(0, 11, 'DESIGNATION'), (12, 57, 'PERSON')]}),
    ('screenplay by,chris terrio', {'entities': [(0, 13, 'DESIGNATION'), (14, 26, 'PERSON')]}),
    ('screenplay by,j. j. abrams', {'entities': [(0, 13, 'DESIGNATION'), (14, 26, 'PERSON')]}),
    ('story by,derek connolly', {'entities': [(0, 8, 'DESIGNATION'), (9, 23, 'PERSON')]}),
    ('story by,colin trevorrow', {'entities': [(0, 8, 'DESIGNATION'), (9, 24, 'PERSON')]}),
    ('story by,j. j. abrams,chris terrio', {'entities': [(0, 8, 'DESIGNATION'), (9, 34, 'PERSON')]}),
    ('based on characters by, george lucas', {'entities': [(0, 22, 'DESIGNATION'), (23, 36, 'PERSON')]}),
    ('music by,john williams', {'entities': [(0, 8, 'DESIGNATION'), (9, 22, 'PERSON')]}),
    ('cinematography,dan mindel', {'entities': [(0, 14, 'DESIGNATION'), (15, 25, 'PERSON')]}),
    ('edited by,maryann brandon', {'entities': [(0, 9, 'DESIGNATION'), (10, 25, 'PERSON')]}),
    ('edited by,stefan grube', {'entities': [(0, 9, 'DESIGNATION'), (10, 22, 'PERSON')]}),
    ('production company,lucasfilm ltd.', {'entities': [(0, 18, 'DESIGNATION'), (19, 33, 'PERSON')]}),
    ('production company,bad robot productions', {'entities': [(0, 18, 'DESIGNATION'), (19, 40, 'PERSON')]}),
    ('distributed by,walt disney studios', {'entities': [(0, 14, 'DESIGNATION'), (15, 34, 'PERSON')]}),
    ('distributed by,motion pictures', {'entities': [(0, 14, 'DESIGNATION'), (15, 30, 'PERSON')]}),
    ('music by,john paesano', {'entities': [(0, 8, 'DESIGNATION'), (9, 21, 'PERSON')]}),
    ('unit proouction manager,james powers', {'entities': [(0, 23, 'DESIGNATION'), (24, 36, 'PERSON')]}),
    ('first assistant director,vincent lascoumes', {'entities': [(0, 24, 'DESIGNATION'), (25, 42, 'PERSON')]}),
    ('first assistant director,jeff jj authors', {'entities': [(0, 24, 'DESIGNATION'), (25, 40, 'PERSON')]}),
    ('second assistant directors,nadia brand', {'entities': [(0, 26, 'DESIGNATION'), (27, 38, 'PERSON')]}),
    ('second assistant directors,joy hoes', {'entities': [(0, 26, 'DESIGNATION'), (27, 35, 'PERSON')]}),
    ('second assistant directors,david klohn', {'entities': [(0, 26, 'DESIGNATION'), (27, 38, 'PERSON')]}),
    ('stunt coordinator,greg kriek', {'entities': [(0, 17, 'DESIGNATION'), (18, 28, 'PERSON')]}),
    ('assistant stunt coordinator,david butler', {'entities': [(0, 27, 'DESIGNATION'), (28, 40, 'PERSON')]}),
    ('fight choreographer,liza scholtz', {'entities': [(0, 19, 'DESIGNATION'), (20, 32, 'PERSON')]}),
    ('unit production manager,susan mcnamara', {'entities': [(0, 23, 'DESIGNATION'), (24, 38, 'PERSON')]}),
    ('unit production manager,david valdes', {'entities': [(0, 23, 'DESIGNATION'), (24, 36, 'PERSON')]}),
    ('first assistant director,brian bettwy', {'entities': [(0, 24, 'DESIGNATION'), (25, 37, 'PERSON')]}),
    ('second assistant director,david vincent rimer', {'entities': [(0, 25, 'DESIGNATION'), (26, 45, 'PERSON')]}),
    ('production accountant,dawn robinette', {'entities': [(0, 21, 'DESIGNATION'), (22, 36, 'PERSON')]}),
    ('stunt coordinator,garrett warren', {'entities': [(0, 17, 'DESIGNATION'), (18, 32, 'PERSON')]}),
    ('weta animation supervisor,michael cozens', {'entities': [(0, 25, 'DESIGNATION'), (26, 40, 'PERSON')]}),
    ('co-production designer,caylah eddleblute', {'entities': [(0, 22, 'DESIGNATION'), (23, 40, 'PERSON')]}),
    ('supervising art director,a.todd holland', {'entities': [(0, 24, 'DESIGNATION'), (25, 39, 'PERSON')]}),
    ('post production supervisor,tom proper', {'entities': [(0, 26, 'DESIGNATION'), (27, 37, 'PERSON')]}),
    ('supervising first assistant editor,jason gaudio', {'entities': [(0, 34, 'DESIGNATION'), (35, 47, 'PERSON')]}),
    ('assistant stunt coordinator,steve brown', {'entities': [(0, 27, 'DESIGNATION'), (28, 39, 'PERSON')]}),
    ('fight choreographer,steve brown', {'entities': [(0, 19, 'DESIGNATION'), (20, 31, 'PERSON')]}),
    ('art directors,leslie mcdonald', {'entities': [(0, 13, 'DESIGNATION'), (14, 29, 'PERSON')]}),
    ('assistant art director,paul alix', {'entities': [(0, 22, 'DESIGNATION'), (23, 32, 'PERSON')]}),
    ('storyboard artists,jim mitchell', {'entities': [(0, 18, 'DESIGNATION'), (19, 31, 'PERSON')]}),
    ('set designers,john berger', {'entities': [(0, 13, 'DESIGNATION'), (14, 25, 'PERSON')]}),
    ('production supervisor,natalie angel', {'entities': [(0, 21, 'DESIGNATION'), (22, 35, 'PERSON')]}),
    ('production coordinator,cynthia streit', {'entities': [(0, 22, 'DESIGNATION'), (23, 37, 'PERSON')]}),
    ('adil el arbi,directed by', {'entities': [(0, 12, 'PERSON'), (13, 24, 'DESIGNATION')]}),
    ('bilall fallah,directed by', {'entities': [(0, 13, 'PERSON'), (14, 25, 'DESIGNATION')]}),
    ('jerry bruckheimer,produced by', {'entities': [(0, 17, 'PERSON'), (18, 29, 'DESIGNATION')]}),
    ('will smith,produced by', {'entities': [(0, 10, 'PERSON'), (11, 22, 'DESIGNATION')]}),
    ('doug belgrad,produced by', {'entities': [(0, 12, 'PERSON'), (13, 24, 'DESIGNATION')]}),
    ('chris bremner,screenplay by', {'entities': [(0, 13, 'PERSON'), (14, 27, 'DESIGNATION')]}),
    ('peter craig,screenplay by', {'entities': [(0, 11, 'PERSON'), (12, 25, 'DESIGNATION')]}),
    ('joe carnahan,screenplay by', {'entities': [(0, 12, 'PERSON'), (13, 26, 'DESIGNATION')]}),
    ('peter craig,story by', {'entities': [(0, 11, 'PERSON'), (12, 20, 'DESIGNATION')]}),
    ('joe carnahan,story by', {'entities': [(0, 12, 'PERSON'), (13, 21, 'DESIGNATION')]}),
    ('characters,based on', {'entities': [(0, 10, 'PERSON'), (11, 19, 'DESIGNATION')]}),
    ('george gallo,based on by', {'entities': [(0, 13, 'PERSON'), (14, 25, 'DESIGNATION')]}),
    ('lorne balfe,music by', {'entities': [(0, 11, 'PERSON'), (12, 20, 'DESIGNATION')]}),
    ('robrecht heyvaert,cinematography', {'entities': [(0, 17, 'PERSON'), (18, 32, 'DESIGNATION')]}),
    ('dan lebental,edited by', {'entities': [(0, 12, 'PERSON'), (13, 22, 'DESIGNATION')]}),
    ('peter mcnulty,edited by', {'entities': [(0, 13, 'PERSON'), (14, 23, 'DESIGNATION')]}),
    ('columbia pictures,production companies', {'entities': [(0, 17, 'PERSON'), (18, 38, 'DESIGNATION')]}),
    ('2.0 entertainment,production companies', {'entities': [(0, 17, 'PERSON'), (18, 38, 'DESIGNATION')]}),
    ('don simpson/jerry,production companies', {'entities': [(0, 17, 'PERSON'), (18, 38, 'DESIGNATION')]}),
    ('bruckheimer films,production companies', {'entities': [(0, 17, 'PERSON'), (18, 38, 'DESIGNATION')]}),
    ('overbrook entertainment,production companies', {'entities': [(0, 23, 'PERSON'), (24, 44, 'DESIGNATION')]}),
    ('sony pictures releasing,distributed by', {'entities': [(0, 23, 'PERSON'), (24, 38, 'DESIGNATION')]}),
    ('j. j. abrams,directed by', {'entities': [(0, 12, 'PERSON'), (13, 24, 'DESIGNATION')]}), (
    'kathleen kennedy,j. j. abrams,michelle rejwan,produced by',
    {'entities': [(0, 16, 'PERSON'), (17, 57, 'DESIGNATION')]}),
    ('chris terrio,screenplay by', {'entities': [(0, 12, 'PERSON'), (13, 26, 'DESIGNATION')]}),
    ('j. j. abrams,screenplay by', {'entities': [(0, 12, 'PERSON'), (13, 26, 'DESIGNATION')]}),
    ('derek connolly,story by', {'entities': [(0, 14, 'PERSON'), (15, 23, 'DESIGNATION')]}),
    ('colin trevorrow,story by', {'entities': [(0, 15, 'PERSON'), (16, 24, 'DESIGNATION')]}),
    ('j. j. abrams,chris terrio,story by', {'entities': [(0, 12, 'PERSON'), (13, 34, 'DESIGNATION')]}),
    ('george lucas,based on characters by', {'entities': [(0, 13, 'PERSON'), (14, 36, 'DESIGNATION')]}),
    ('john williams,music by', {'entities': [(0, 13, 'PERSON'), (14, 22, 'DESIGNATION')]}),
    ('dan mindel,cinematography', {'entities': [(0, 10, 'PERSON'), (11, 25, 'DESIGNATION')]}),
    ('maryann brandon,edited by', {'entities': [(0, 15, 'PERSON'), (16, 25, 'DESIGNATION')]}),
    ('stefan grube,edited by', {'entities': [(0, 12, 'PERSON'), (13, 22, 'DESIGNATION')]}),
    ('lucasfilm ltd.,production company', {'entities': [(0, 14, 'PERSON'), (15, 33, 'DESIGNATION')]}),
    ('bad robot productions,production company', {'entities': [(0, 21, 'PERSON'), (22, 40, 'DESIGNATION')]}),
    ('walt disney studios,distributed by', {'entities': [(0, 19, 'PERSON'), (20, 34, 'DESIGNATION')]}),
    ('motion pictures,distributed by', {'entities': [(0, 15, 'PERSON'), (16, 30, 'DESIGNATION')]}),
    ('john paesano,music by', {'entities': [(0, 12, 'PERSON'), (13, 21, 'DESIGNATION')]}),
    ('james powers,unit proouction manager', {'entities': [(0, 12, 'PERSON'), (13, 36, 'DESIGNATION')]}),
    ('vincent lascoumes,first assistant director', {'entities': [(0, 17, 'PERSON'), (18, 42, 'DESIGNATION')]}),
    ('jeff jj authors,first assistant director', {'entities': [(0, 15, 'PERSON'), (16, 40, 'DESIGNATION')]}),
    ('nadia brand,second assistant directors', {'entities': [(0, 11, 'PERSON'), (12, 38, 'DESIGNATION')]}),
    ('joy hoes,second assistant directors', {'entities': [(0, 8, 'PERSON'), (9, 35, 'DESIGNATION')]}),
    ('david klohn,second assistant directors', {'entities': [(0, 11, 'PERSON'), (12, 38, 'DESIGNATION')]}),
    ('greg kriek,stunt coordinator', {'entities': [(0, 10, 'PERSON'), (11, 28, 'DESIGNATION')]}),
    ('david butler,assistant stunt coordinator', {'entities': [(0, 12, 'PERSON'), (13, 40, 'DESIGNATION')]}),
    ('liza scholtz,fight choreographer', {'entities': [(0, 12, 'PERSON'), (13, 32, 'DESIGNATION')]}),
    ('susan mcnamara,unit production manager', {'entities': [(0, 14, 'PERSON'), (15, 38, 'DESIGNATION')]}),
    ('david valdes,unit production manager', {'entities': [(0, 12, 'PERSON'), (13, 36, 'DESIGNATION')]}),
    ('brian bettwy,first assistant director', {'entities': [(0, 12, 'PERSON'), (13, 37, 'DESIGNATION')]}),
    ('david vincent rimer,second assistant director', {'entities': [(0, 19, 'PERSON'), (20, 45, 'DESIGNATION')]}),
    ('dawn robinette,production accountant', {'entities': [(0, 14, 'PERSON'), (15, 36, 'DESIGNATION')]}),
    ('garrett warren,stunt coordinator', {'entities': [(0, 14, 'PERSON'), (15, 32, 'DESIGNATION')]}),
    ('michael cozens,weta animation supervisor', {'entities': [(0, 14, 'PERSON'), (15, 40, 'DESIGNATION')]}),
    ('caylah eddleblute,co-production designer', {'entities': [(0, 17, 'PERSON'), (18, 40, 'DESIGNATION')]}),
    ('a.todd holland,supervising art director', {'entities': [(0, 14, 'PERSON'), (15, 39, 'DESIGNATION')]}),
    ('tom proper,post production supervisor', {'entities': [(0, 10, 'PERSON'), (11, 37, 'DESIGNATION')]}),
    ('jason gaudio,supervising first assistant editor', {'entities': [(0, 12, 'PERSON'), (13, 47, 'DESIGNATION')]}),
    ('steve brown,assistant stunt coordinator', {'entities': [(0, 11, 'PERSON'), (12, 39, 'DESIGNATION')]}),
    ('steve brown,fight choreographer', {'entities': [(0, 11, 'PERSON'), (12, 31, 'DESIGNATION')]}),
    ('leslie mcdonald,art directors', {'entities': [(0, 15, 'PERSON'), (16, 29, 'DESIGNATION')]}),
    ('paul alix,assistant art director', {'entities': [(0, 9, 'PERSON'), (10, 32, 'DESIGNATION')]}),
    ('jim mitchell,storyboard artists', {'entities': [(0, 12, 'PERSON'), (13, 31, 'DESIGNATION')]}),
    ('john berger,set designers', {'entities': [(0, 11, 'PERSON'), (12, 25, 'DESIGNATION')]}),
    ('natalie angel,production supervisor', {'entities': [(0, 13, 'PERSON'), (14, 35, 'DESIGNATION')]}),
    ('cynthia streit,production coordinator', {'entities': [(0, 14, 'PERSON'), (15, 37, 'DESIGNATION')]}),
    ('directed by,robert rodriguez', {'entities': [(0, 11, 'DESIGNATION'), (12, 28, 'PERSON')]}), (
    'produced by,james cameron,jon landau',
    {'entities': [(0, 11, 'DESIGNATION'), (12, 25, 'PERSON'), (26, 36, 'PERSON')]}), (
    'screenplay by,james cameron,laeta kalogridis',
    {'entities': [(0, 13, 'DESIGNATION'), (14, 27, 'PERSON'), (28, 44, 'PERSON')]}), (
    'edited by,stephen e. rivkin,ian silverstein',
    {'entities': [(0, 9, 'DESIGNATION'), (10, 27, 'PERSON'), (28, 43, 'PERSON')]}), (
    'associate producers,rick porras,steven j. boyd,rob lowe,jake busey', {
        'entities': [(0, 19, 'DESIGNATION'), (20, 31, 'PERSON'), (32, 46, 'PERSON'), (47, 55, 'PERSON'),
                     (56, 66, 'PERSON')]}), ('screenplay by,james v. hart and michael goldenberg', {
        'entities': [(0, 13, 'DESIGNATION'), (14, 27, 'PERSON'), (32, 50, 'PERSON')]}),
    ('a robert zemeckis film', {'entities': [(2, 17, 'PERSON')]}),
    ('based on the novel by,carl sagan', {'entities': [(0, 21, 'DESIGNATION'), (22, 32, 'PERSON')]}),
    ('director of photography,don burgess', {'entities': [(0, 23, 'DESIGNATION'), (24, 35, 'PERSON')]}),
    ('production designer,ed verreaux', {'entities': [(0, 19, 'DESIGNATION'), (20, 31, 'PERSON')]}),
    ('edited by,arthur schmidt', {'entities': [(0, 9, 'DESIGNATION'), (10, 24, 'PERSON')]}),
    ('music by,alan silvestri', {'entities': [(0, 8, 'DESIGNATION'), (9, 23, 'PERSON')]}),
    ('directed by,robert zemeckis', {'entities': [(0, 11, 'DESIGNATION'), (12, 27, 'PERSON')]}),
    ('senior visual effects supervisor,ken ralston', {'entities': [(0, 32, 'DESIGNATION'), (33, 44, 'PERSON')]}),
    ('second assistant director,cellin gluck', {'entities': [(0, 25, 'DESIGNATION'), (26, 38, 'PERSON')]}),
    ('second second assistant director,darin rivetti', {'entities': [(0, 32, 'DESIGNATION'), (33, 46, 'PERSON')]}),
    ('costumes designed by,joanna johnston', {'entities': [(0, 20, 'DESIGNATION'), (21, 36, 'PERSON')]}),
    ('casting by,victoria burrows', {'entities': [(0, 10, 'DESIGNATION'), (11, 27, 'PERSON')]}),
    ('co-producers,carl sagan and ann druyan', {'entities': [(0, 12, 'DESIGNATION'), (13, 38, 'PERSON')]}),
    ('unit production manager,joan bradshaw', {'entities': [(0, 23, 'DESIGNATION'), (24, 37, 'PERSON')]}),
    ('first assistant director,bruce moriarty', {'entities': [(0, 24, 'DESIGNATION'), (25, 39, 'PERSON')]}),
    ('unit production manager,cherylanne martin', {'entities': [(0, 23, 'DESIGNATION'), (24, 41, 'PERSON')]}), (
    'art directors,lawrence a. hubbs,bruce crone',
    {'entities': [(0, 13, 'DESIGNATION'), (14, 31, 'PERSON'), (32, 43, 'PERSON')]}),
    ('video graphics supervisor,ian kelly', {'entities': [(0, 25, 'DESIGNATION'), (26, 35, 'PERSON')]}), (
    'assistan editors,fred vitale,jana gold,mathew schmidt,sean menzies', {
        'entities': [(0, 16, 'DESIGNATION'), (17, 28, 'PERSON'), (29, 38, 'PERSON'), (39, 53, 'PERSON'),
                     (54, 66, 'PERSON')]}), ('directed by,adil el arbi,bilall fallah', {
        'entities': [(0, 11, 'DESIGNATION'), (12, 24, 'PERSON'), (25, 38, 'PERSON')]}), (
    'produced by,jerry bruckheimer,will smith,doug belgrad',
    {'entities': [(0, 11, 'DESIGNATION'), (12, 29, 'PERSON'), (30, 40, 'PERSON'), (41, 53, 'PERSON')]}), (
    'stunt riggers,uli ritcher,niklas kinzel',
    {'entities': [(0, 13, 'DESIGNATION'), (14, 25, 'PERSON'), (26, 39, 'PERSON')]}), (
    'co-producers christoph fisser,henning molfenter,charlie woebcken',
    {'entities': [(0, 12, 'DESIGNATION'), (13, 29, 'PERSON'), (30, 47, 'PERSON'), (48, 64, 'PERSON')]}), (
    'associate producers,ari handel,evan ginzburg',
    {'entities': [(0, 19, 'DESIGNATION'), (20, 30, 'PERSON'), (31, 44, 'PERSON')]}), (
    'sound supervision and design,jacob ribicoff,brian emrich',
    {'entities': [(0, 28, 'DESIGNATION'), (29, 43, 'PERSON'), (44, 56, 'PERSON')]}), (
    'first assistant camera,jeff dutemple,malcom purnell',
    {'entities': [(0, 22, 'DESIGNATION'), (23, 36, 'PERSON'), (37, 51, 'PERSON')]}), (
    'second assistant camera,daniel wiener,travis cadalzo',
    {'entities': [(0, 23, 'DESIGNATION'), (24, 37, 'PERSON'), (38, 52, 'PERSON')]}), (
    'genny operators,robert gurgo,william hines',
    {'entities': [(0, 15, 'DESIGNATION'), (16, 28, 'PERSON'), (29, 42, 'PERSON')]}), (
    'rigging gaffers,michael gallart,timothy healy',
    {'entities': [(0, 15, 'DESIGNATION'), (16, 31, 'PERSON'), (32, 45, 'PERSON')]}),
    ('unit production manager,susan mcnamara', {'entities': [(0, 23, 'DESIGNATION'), (24, 38, 'PERSON')]}),
    ('unit production manager,david valdes', {'entities': [(0, 23, 'DESIGNATION'), (24, 36, 'PERSON')]}),
    ('first assistant director,brian bettwy', {'entities': [(0, 24, 'DESIGNATION'), (25, 37, 'PERSON')]}),
    ('unit production manager,cristen carr strubbe', {'entities': [(0, 23, 'DESIGNATION'), (24, 44, 'PERSON')]}),
    ('first assistant director,sergio mimica-gezzan', {'entities': [(0, 24, 'DESIGNATION'), (25, 45, 'PERSON')]}),
    ('second assistant director,david h. venghaus jr.', {'entities': [(0, 25, 'DESIGNATION'), (26, 47, 'PERSON')]}),
    ('driving instructor,danica patrick', {'entities': [(0, 18, 'DESIGNATION'), (19, 33, 'PERSON')]}),
    ('fight instructor,ronda rousey', {'entities': [(0, 16, 'DESIGNATION'), (17, 29, 'PERSON')]}),
    ('bomb instructor,laverene cox', {'entities': [(0, 15, 'DESIGNATION'), (16, 28, 'PERSON')]}),
    ('stunt coordinator,florian hotz', {'entities': [(0, 17, 'DESIGNATION'), (18, 30, 'PERSON')]}),
    ('assistant stunt coordinator,sandra barger', {'entities': [(0, 27, 'DESIGNATION'), (28, 41, 'PERSON')]}),
    ('fight choreographer,tolga degirmen', {'entities': [(0, 19, 'DESIGNATION'), (20, 34, 'PERSON')]}),
    ('head stunt rigger,alexander magerl', {'entities': [(0, 17, 'DESIGNATION'), (18, 34, 'PERSON')]}),
    ('unit production manager,arno neubauer', {'entities': [(0, 23, 'DESIGNATION'), (24, 37, 'PERSON')]}),
    ('first assistant director,alex oakley', {'entities': [(0, 24, 'DESIGNATION'), (25, 36, 'PERSON')]}),
    ('associate producer,alex oakley', {'entities': [(0, 18, 'DESIGNATION'), (19, 30, 'PERSON')]}),
    ('second assistant director,james manning', {'entities': [(0, 25, 'DESIGNATION'), (26, 39, 'PERSON')]}),
    ('visual effects supervisor,karen heston', {'entities': [(0, 25, 'DESIGNATION'), (26, 38, 'PERSON')]}),
    ('unit production manager,jennifer roth', {'entities': [(0, 23, 'DESIGNATION'), (24, 37, 'PERSON')]}),
    ('first assistant director,richard graves', {'entities': [(0, 24, 'DESIGNATION'), (25, 39, 'PERSON')]}),
    ('second assistant director,brendan walsh', {'entities': [(0, 25, 'DESIGNATION'), (26, 39, 'PERSON')]}),
    ('stunt coordinator,douglas crosby', {'entities': [(0, 17, 'DESIGNATION'), (18, 32, 'PERSON')]}),
    ('production supervisor,alexis arnold', {'entities': [(0, 21, 'DESIGNATION'), (22, 35, 'PERSON')]}),
    ('script supervisor,anthony pettine', {'entities': [(0, 17, 'DESIGNATION'), (18, 33, 'PERSON')]}),
    ('post production supervisor,colleen bachman', {'entities': [(0, 26, 'DESIGNATION'), (27, 42, 'PERSON')]}),
    ('camera operator,peter nolan', {'entities': [(0, 15, 'DESIGNATION'), (16, 27, 'PERSON')]}),
    ('stills photographer,niko tavernise', {'entities': [(0, 19, 'DESIGNATION'), (20, 34, 'PERSON')]}),
    ('24 frame playback,mike sime', {'entities': [(0, 17, 'DESIGNATION'), (18, 27, 'PERSON')]}),
    ('camera intern,aurore guyot', {'entities': [(0, 13, 'DESIGNATION'), (14, 26, 'PERSON')]}),
    ('gaffer,david skutch', {'entities': [(0, 6, 'DESIGNATION'), (7, 19, 'PERSON')]}),
    ('best boy, jarad molkenthin', {'entities': [(0, 8, 'DESIGNATION'), (9, 26, 'PERSON')]}),
    ('rigging electric,james temme', {'entities': [(0, 16, 'DESIGNATION'), (17, 28, 'PERSON')]}),
    ('theatrical lighting technician,micheal gallart', {'entities': [(0, 30, 'DESIGNATION'), (31, 46, 'PERSON')]}), (
    'assistant hair stylists,jeffrey rebelo,diana sikes',
    {'entities': [(0, 23, 'DESIGNATION'), (24, 38, 'PERSON'), (39, 50, 'PERSON')]}), (
    'assistant location managers,nate braeuer,lauren fritz',
    {'entities': [(0, 27, 'DESIGNATION'), (28, 40, 'PERSON'), (41, 53, 'PERSON')]}), (
    're-recording mixers,dominick tavella,jacob ribicoff',
    {'entities': [(0, 19, 'DESIGNATION'), (20, 36, 'PERSON'), (37, 51, 'PERSON')]}), (
    'eliza paley,william sweeney,sound effects editors',
    {'entities': [(0, 11, 'PERSON'), (12, 27, 'PERSON'), (28, 49, 'DESIGNATION')]}),
    ('chris skutch,key grip', {'entities': [(0, 12, 'PERSON'), (13, 21, 'DESIGNATION')]}),
    ('john haligan,best boy', {'entities': [(0, 12, 'PERSON'), (13, 21, 'DESIGNATION')]}),
    ("ben d'andrea,dolly grip", {'entities': [(0, 12, 'PERSON'), (13, 23, 'DESIGNATION')]}),
    ('ruark behan,grip', {'entities': [(0, 11, 'PERSON'), (12, 16, 'DESIGNATION')]}),
    ('graham klatt,key rigging grip', {'entities': [(0, 12, 'PERSON'), (13, 29, 'DESIGNATION')]}),
    ('reid kelly,boy rigging grip', {'entities': [(0, 10, 'PERSON'), (11, 27, 'DESIGNATION')]}),
    ('theo sena,set decorator', {'entities': [(0, 9, 'PERSON'), (10, 23, 'DESIGNATION')]}),
    ('scott gagnon,leadman', {'entities': [(0, 12, 'PERSON'), (13, 20, 'DESIGNATION')]}),
    ('tim rossiter,additional leadman', {'entities': [(0, 12, 'PERSON'), (13, 31, 'DESIGNATION')]}),
    ('jonathan unger,set dresser', {'entities': [(0, 14, 'PERSON'), (15, 26, 'DESIGNATION')]}),
    ('robin koenig,on-set dresser', {'entities': [(0, 12, 'PERSON'), (13, 27, 'DESIGNATION')]}),
    ('travis child,sceanic', {'entities': [(0, 12, 'PERSON'), (13, 20, 'DESIGNATION')]}),
    ('jeff butcher,prop master', {'entities': [(0, 12, 'PERSON'), (13, 24, 'DESIGNATION')]}),
    ('daniel fisher,prop master', {'entities': [(0, 13, 'PERSON'), (14, 25, 'DESIGNATION')]}),
    ('eoin lambe,3rd prop assistant', {'entities': [(0, 10, 'PERSON'), (11, 29, 'DESIGNATION')]}),
    ('matthew munn,art director', {'entities': [(0, 12, 'PERSON'), (13, 25, 'DESIGNATION')]}),
    ('selina van den brink,art department coordinator', {'entities': [(0, 20, 'PERSON'), (21, 47, 'DESIGNATION')]}),
    ('shane ingersoll,art production assistant', {'entities': [(0, 15, 'PERSON'), (16, 40, 'DESIGNATION')]}),
    ('craig hench,art department intern', {'entities': [(0, 11, 'PERSON'), (12, 33, 'DESIGNATION')]}),
    ('lenore pemberton,assistant costume designer', {'entities': [(0, 16, 'PERSON'), (17, 43, 'DESIGNATION')]}),
    ('steffany bernstein,wardrobe supervisor', {'entities': [(0, 18, 'PERSON'), (19, 38, 'DESIGNATION')]}),
    ('nicci schinman,set costumer', {'entities': [(0, 14, 'PERSON'), (15, 27, 'DESIGNATION')]}),
    ('cara czekanski,additional costumer', {'entities': [(0, 14, 'PERSON'), (15, 34, 'DESIGNATION')]}),
    ('estella simmons,alterations', {'entities': [(0, 15, 'PERSON'), (16, 27, 'DESIGNATION')]}),
    ('rena bussinger,costume intern', {'entities': [(0, 14, 'PERSON'), (15, 29, 'DESIGNATION')]}),
    ('monte cristo,ram belt buckle design', {'entities': [(0, 12, 'PERSON'), (13, 35, 'DESIGNATION')]}),
    ('judy chin,key makeup', {'entities': [(0, 9, 'PERSON'), (10, 20, 'DESIGNATION')]}),
    ('eve morrow,makeup artist', {'entities': [(0, 10, 'PERSON'), (11, 24, 'DESIGNATION')]}),
    ('margie durand,assistant makeup artist', {'entities': [(0, 13, 'PERSON'), (14, 37, 'DESIGNATION')]}),
    ('mandy lyons,key hair stylist', {'entities': [(0, 11, 'PERSON'), (12, 28, 'DESIGNATION')]}),
    ('mike marino,prosthetic makeup designer', {'entities': [(0, 11, 'PERSON'), (12, 38, 'DESIGNATION')]}),
    ('hayes vilandry,lab supervisor', {'entities': [(0, 14, 'PERSON'), (15, 29, 'DESIGNATION')]}),
    ('drew jiritano,special effects', {'entities': [(0, 13, 'PERSON'), (14, 29, 'DESIGNATION')]}),
    ('ken ishii,sound mixer', {'entities': [(0, 9, 'PERSON'), (10, 21, 'DESIGNATION')]}),
    ('anguibe guindo,boom operator', {'entities': [(0, 14, 'PERSON'), (15, 28, 'DESIGNATION')]}),
    ('joe origlieri,additional sound b-unit', {'entities': [(0, 13, 'PERSON'), (14, 37, 'DESIGNATION')]}),
    ('joe origlieri,sound utility', {'entities': [(0, 13, 'PERSON'), (14, 27, 'DESIGNATION')]}),
    ('ryan smith,location manager', {'entities': [(0, 10, 'PERSON'), (11, 27, 'DESIGNATION')]}),
    ('abi jackson,key location assistant', {'entities': [(0, 11, 'PERSON'), (12, 34, 'DESIGNATION')]}),
    ('dan pollack,location scout', {'entities': [(0, 11, 'PERSON'), (12, 26, 'DESIGNATION')]}),
    ('zach thrun,unit pa', {'entities': [(0, 10, 'PERSON'), (11, 18, 'DESIGNATION')]}),
    ('henry winnik,locations interns', {'entities': [(0, 12, 'PERSON'), (13, 30, 'DESIGNATION')]}),
    ('kenneth wachtel,first assistant editor', {'entities': [(0, 15, 'PERSON'), (16, 38, 'DESIGNATION')]}),
    ('sebastian ischer,additional assistant editor', {'entities': [(0, 16, 'PERSON'), (17, 44, 'DESIGNATION')]}),
    ('barry blaschke,post production pa', {'entities': [(0, 14, 'PERSON'), (15, 33, 'DESIGNATION')]}),
    ('sound one,post production sound services', {'entities': [(0, 9, 'PERSON'), (10, 40, 'DESIGNATION')]}),
    ('tony martinez,adr supervisor', {'entities': [(0, 13, 'PERSON'), (14, 28, 'DESIGNATION')]}),
    ('tony martinez,dialoague supervisor', {'entities': [(0, 13, 'PERSON'), (14, 34, 'DESIGNATION')]}),
    ('daniel korintus,dialogue editor', {'entities': [(0, 15, 'PERSON'), (16, 31, 'DESIGNATION')]}),
    ('stuart stanley,foley editor', {'entities': [(0, 14, 'PERSON'), (15, 27, 'DESIGNATION')]}),
    ('eric strausser,assistant sound editor', {'entities': [(0, 14, 'PERSON'), (15, 37, 'DESIGNATION')]}),
    ('dror gescheit,machine room operator', {'entities': [(0, 13, 'PERSON'), (14, 35, 'DESIGNATION')]}),
    ('david boulton,adr mixer', {'entities': [(0, 13, 'PERSON'), (14, 23, 'DESIGNATION')]}),
    ('mike howells,adr machine room operator', {'entities': [(0, 12, 'PERSON'), (13, 38, 'DESIGNATION')]}),
    ('ryan collison,foley mixer', {'entities': [(0, 13, 'PERSON'), (14, 25, 'DESIGNATION')]}),
    ('jay peck,foley walker', {'entities': [(0, 8, 'PERSON'), (9, 21, 'DESIGNATION')]}),
    ('eric vierhaus,dolby sound consultant', {'entities': [(0, 13, 'PERSON'), (14, 36, 'DESIGNATION')]}),
    ('dan evans farkas,music editor', {'entities': [(0, 16, 'PERSON'), (17, 29, 'DESIGNATION')]}),
    ('Is Taiwan a good place to live?', {'entities':[(3, 9, 'GPE')]}),
    ('Is Pakistan a good place to live?', {'entities':[(3, 11, 'GPE')]}),
    ('I hate Uganda as it is very hot over there', {'entities':[(7, 13, 'GPE')]}),
    ('England is a small place', {'entities':[(0, 7, 'GPE')]}),
    ('I used to live in Africa and then after 2 years shifted to America', {'entities':[(18, 24, 'LOC'),(40, 47, 'DATE'), (59, 66, 'GPE')]}),
    ('My native place is India', {'entities':[(19, 24, 'GPE')]}),
    ('My house is near the river bank overlooking Kolkata', {'entities':[(44, 51, 'GPE')]}),
    ('Switzerland has a very backdrop thereby it is a good tourist attraction', {'entities':[(0, 11, 'GPE')]}),
    ('He has houses in England,France,Dubai,Los Angeles,Meerut,Rome among other places', {'entities':[(17, 24, 'GPE'), (25, 31, 'GPE'), (32, 37, 'GPE'), (38, 49, 'GPE'), (50, 56, 'GPE'), (57, 61, 'GPE')]}),
    ('NOKIA had a very good sale this year', {'entities':[(0, 5, 'ORG'), (27, 36, 'DATE')]}),
    ('His brother holds 30% of the total shares', {'entities':[(18, 21, 'PERCENT')]}),
    ('He can speak Hindi', {'entities':[(13, 18, 'LANGUAGE')]}),
    ('We speak English', {'entities':[(9, 16, 'LANGUAGE')]}),
    ('He is the first son of James', {'entities':[(10, 15, 'ORDINAL'), (23, 28, 'PERSON')]}),
    ('I am second youngest', {'entities':[(5, 11, 'ORDINAL')]}),
    ('He is the tallest of them all by atleast 20%', {'entities':[(41, 44, 'PERCENT')]}),
    ('I owe him 80000 rupees', {'entities':[(10, 21, 'MONEY')]}),('Adil El Arbi Directed by', {'entities': [(0, 12, 'PERSON'), (13, 24, 'DESIGNATION')]}), ('Bilall Fallah Directed by', {'entities': [(0, 13, 'PERSON'), (14, 25, 'DESIGNATION')]}), ('Jerry Bruckheimer Produced by', {'entities': [(0, 17, 'PERSON'), (18, 29, 'DESIGNATION')]}), ('Will Smith Produced by', {'entities': [(0, 10, 'PERSON'), (11, 22, 'DESIGNATION')]}), ('Doug Belgrad Produced by', {'entities': [(0, 12, 'PERSON'), (13, 24, 'DESIGNATION')]}), ('Chris Bremner Screenplay by', {'entities': [(0, 13, 'PERSON'), (14, 27, 'DESIGNATION')]}), ('Peter Craig Screenplay by', {'entities': [(0, 11, 'PERSON'), (12, 25, 'DESIGNATION')]}), ('Joe Carnahan Screenplay by', {'entities': [(0, 12, 'PERSON'), (13, 26, 'DESIGNATION')]}), ('Peter Craig Story by', {'entities': [(0, 11, 'PERSON'), (12, 20, 'DESIGNATION')]}), ('Joe Carnahan Story by', {'entities': [(0, 12, 'PERSON'), (13, 21, 'DESIGNATION')]}), ('Characters Based on', {'entities': [(0, 10, 'PERSON'), (11, 19, 'DESIGNATION')]}), ('George Gallo Based on by', {'entities': [(0, 13, 'PERSON'), (14, 25, 'DESIGNATION')]}), ('Lorne Balfe Music by', {'entities': [(0, 11, 'PERSON'), (12, 20, 'DESIGNATION')]}), ('Robrecht Heyvaert Cinematography', {'entities': [(0, 17, 'PERSON'), (18, 32, 'DESIGNATION')]}), ('Dan Lebental Edited by', {'entities': [(0, 12, 'PERSON'), (13, 22, 'DESIGNATION')]}), ('Peter McNulty Edited by', {'entities': [(0, 13, 'PERSON'), (14, 23, 'DESIGNATION')]}), ('Columbia Pictures Production companies', {'entities': [(0, 17, 'PERSON'), (18, 38, 'DESIGNATION')]}), ('2.0 Entertainment Production companies', {'entities': [(0, 17, 'PERSON'), (18, 38, 'DESIGNATION')]}), ('Don Simpson/Jerry Production companies', {'entities': [(0, 17, 'PERSON'), (18, 38, 'DESIGNATION')]}), ('Bruckheimer Films Production companies', {'entities': [(0, 17, 'PERSON'), (18, 38, 'DESIGNATION')]}), ('Overbrook Entertainment Production companies', {'entities': [(0, 23, 'PERSON'), (24, 44, 'DESIGNATION')]}), ('Sony Pictures Releasing Distributed by', {'entities': [(0, 23, 'PERSON'), (24, 38, 'DESIGNATION')]}), ('J. J. Abrams Directed by', {'entities': [(0, 12, 'PERSON'), (13, 24, 'DESIGNATION')]}), ('Kathleen Kennedy J. J. Abrams,Michelle Rejwan,Produced by', {'entities': [(0, 16, 'PERSON'), (17, 57, 'DESIGNATION')]}), ('Chris Terrio Screenplay by', {'entities': [(0, 12, 'PERSON'), (13, 26, 'DESIGNATION')]}), ('J. J. Abrams Screenplay by', {'entities': [(0, 12, 'PERSON'), (13, 26, 'DESIGNATION')]}), ('Derek Connolly Story by', {'entities': [(0, 14, 'PERSON'), (15, 23, 'DESIGNATION')]}), ('Colin Trevorrow Story by', {'entities': [(0, 15, 'PERSON'), (16, 24, 'DESIGNATION')]}), ('J. J. Abrams Chris Terrio Story by', {'entities': [(0, 12, 'PERSON'), (13, 25, 'PERSON'),(26, 34, 'DESIGNATION')]}), ('George Lucas Based on Characters by', {'entities': [(0, 13, 'PERSON'), (14, 36, 'DESIGNATION')]}), ('John Williams Music by', {'entities': [(0, 13, 'PERSON'), (14, 22, 'DESIGNATION')]}), ('Dan Mindel Cinematography', {'entities': [(0, 10, 'PERSON'), (11, 25, 'DESIGNATION')]}), ('Maryann Brandon Edited by', {'entities': [(0, 15, 'PERSON'), (16, 25, 'DESIGNATION')]}), ('Stefan Grube Edited by', {'entities': [(0, 12, 'PERSON'), (13, 22, 'DESIGNATION')]}), ('Lucasfilm Ltd. Production company', {'entities': [(0, 14, 'PERSON'), (15, 33, 'DESIGNATION')]}), ('Bad Robot Productions Production company', {'entities': [(0, 21, 'PERSON'), (22, 40, 'DESIGNATION')]}), ('Walt Disney Studios Distributed by', {'entities': [(0, 19, 'PERSON'), (20, 34, 'DESIGNATION')]}), ('Motion Pictures Distributed by', {'entities': [(0, 15, 'PERSON'), (16, 30, 'DESIGNATION')]}), ('JOHN PAESANO MUSIC By', {'entities': [(0, 12, 'PERSON'), (13, 21, 'DESIGNATION')]}), ('JAMES POWERS UNIT PROOUCTION MANAGER', {'entities': [(0, 12, 'PERSON'), (13, 36, 'DESIGNATION')]}), ('VINCENT LASCOUMES FIRST ASSISTANT DIRECTOR', {'entities': [(0, 17, 'PERSON'), (18, 42, 'DESIGNATION')]}), ('JEFF JJ AUTHORS FIRST ASSISTANT DIRECTOR', {'entities': [(0, 15, 'PERSON'), (16, 40, 'DESIGNATION')]}), ('NADIA BRAND SECOND ASSISTANT DIRECTORS', {'entities': [(0, 11, 'PERSON'), (12, 38, 'DESIGNATION')]}), ('JOY HOES SECOND ASSISTANT DIRECTORS', {'entities': [(0, 8, 'PERSON'), (9, 35, 'DESIGNATION')]}), ('DAVID KLOHN SECOND ASSISTANT DIRECTORS', {'entities': [(0, 11, 'PERSON'), (12, 38, 'DESIGNATION')]}), ('GREG KRIEK STUNT COORDINATOR', {'entities': [(0, 10, 'PERSON'), (11, 28, 'DESIGNATION')]}), ('DAVID BUTLER ASSISTANT STUNT COORDINATOR', {'entities': [(0, 12, 'PERSON'), (13, 40, 'DESIGNATION')]}), ('LIZA SCHOLTZ FIGHT CHOREOGRAPHER', {'entities': [(0, 12, 'PERSON'), (13, 32, 'DESIGNATION')]}), ('SUSAN McNAMARA Unit Production Manager', {'entities': [(0, 14, 'PERSON'), (15, 38, 'DESIGNATION')]}), ('DAVID VALDES Unit Production Manager', {'entities': [(0, 12, 'PERSON'), (13, 36, 'DESIGNATION')]}), ('BRIAN BETTWY First Assistant Director', {'entities': [(0, 12, 'PERSON'), (13, 37, 'DESIGNATION')]}), ('DAVID VINCENT RIMER Second Assistant Director', {'entities': [(0, 19, 'PERSON'), (20, 45, 'DESIGNATION')]}), ('DAWN ROBINETTE Production Accountant', {'entities': [(0, 14, 'PERSON'), (15, 36, 'DESIGNATION')]}), ('GARRETT WARREN Stunt Coordinator', {'entities': [(0, 14, 'PERSON'), (15, 32, 'DESIGNATION')]}), ('MICHAEL COZENS WETA Animation Supervisor', {'entities': [(0, 14, 'PERSON'), (15, 40, 'DESIGNATION')]}), ('CAYLAH EDDLEBLUTE Co-Production Designer', {'entities': [(0, 17, 'PERSON'), (18, 40, 'DESIGNATION')]}), ('A.TODD HOLLAND Supervising Art Director', {'entities': [(0, 14, 'PERSON'), (15, 39, 'DESIGNATION')]}), ('TOM PROPER Post Production Supervisor', {'entities': [(0, 10, 'PERSON'), (11, 37, 'DESIGNATION')]}), ('JASON GAUDIO Supervising First Assistant Editor', {'entities': [(0, 12, 'PERSON'), (13, 47, 'DESIGNATION')]}), ('STEVE BROWN Assistant Stunt Coordinator', {'entities': [(0, 11, 'PERSON'), (12, 39, 'DESIGNATION')]}), ('STEVE BROWN Fight Choreographer', {'entities': [(0, 11, 'PERSON'), (12, 31, 'DESIGNATION')]}), ('LESLIE McDONALD Art Directors', {'entities': [(0, 15, 'PERSON'), (16, 29, 'DESIGNATION')]}), ('PAUL ALIX Assistant Art Director', {'entities': [(0, 9, 'PERSON'), (10, 32, 'DESIGNATION')]}), ('JIM MITCHELL Storyboard Artists', {'entities': [(0, 12, 'PERSON'), (13, 31, 'DESIGNATION')]}), ('JOHN BERGER Set Designers', {'entities': [(0, 11, 'PERSON'), (12, 25, 'DESIGNATION')]}), ('NATALIE ANGEL Production Supervisor', {'entities': [(0, 13, 'PERSON'), (14, 35, 'DESIGNATION')]}), ('CYNTHIA STREIT Production Coordinator', {'entities': [(0, 14, 'PERSON'), (15, 37, 'DESIGNATION')]}),('adil el arbi directed by', {'entities': [(0, 12, 'PERSON'), (13, 24, 'DESIGNATION')]}), ('bilall fallah directed by', {'entities': [(0, 13, 'PERSON'), (14, 25, 'DESIGNATION')]}), ('jerry bruckheimer produced by', {'entities': [(0, 17, 'PERSON'), (18, 29, 'DESIGNATION')]}), ('will smith produced by', {'entities': [(0, 10, 'PERSON'), (11, 22, 'DESIGNATION')]}), ('doug belgrad produced by', {'entities': [(0, 12, 'PERSON'), (13, 24, 'DESIGNATION')]}), ('chris bremner screenplay by', {'entities': [(0, 13, 'PERSON'), (14, 27, 'DESIGNATION')]}), ('peter craig screenplay by', {'entities': [(0, 11, 'PERSON'), (12, 25, 'DESIGNATION')]}), ('joe carnahan screenplay by', {'entities': [(0, 12, 'PERSON'), (13, 26, 'DESIGNATION')]}), ('peter craig story by', {'entities': [(0, 11, 'PERSON'), (12, 20, 'DESIGNATION')]}), ('joe carnahan story by', {'entities': [(0, 12, 'PERSON'), (13, 21, 'DESIGNATION')]}), ('characters based on', {'entities': [(0, 10, 'PERSON'), (11, 19, 'DESIGNATION')]}), ('george gallo based on by', {'entities': [(0, 13, 'PERSON'), (14, 25, 'DESIGNATION')]}), ('lorne balfe music by', {'entities': [(0, 11, 'PERSON'), (12, 20, 'DESIGNATION')]}), ('robrecht heyvaert cinematography', {'entities': [(0, 17, 'PERSON'), (18, 32, 'DESIGNATION')]}), ('dan lebental edited by', {'entities': [(0, 12, 'PERSON'), (13, 22, 'DESIGNATION')]}), ('peter mcnulty edited by', {'entities': [(0, 13, 'PERSON'), (14, 23, 'DESIGNATION')]}), ('columbia pictures production companies', {'entities': [(0, 17, 'PERSON'), (18, 38, 'DESIGNATION')]}), ('2.0 entertainment production companies', {'entities': [(0, 17, 'PERSON'), (18, 38, 'DESIGNATION')]}), ('don simpson/jerry production companies', {'entities': [(0, 17, 'PERSON'), (18, 38, 'DESIGNATION')]}), ('bruckheimer films production companies', {'entities': [(0, 17, 'PERSON'), (18, 38, 'DESIGNATION')]}), ('overbrook entertainment production companies', {'entities': [(0, 23, 'PERSON'), (24, 44, 'DESIGNATION')]}), ('sony pictures releasing distributed by', {'entities': [(0, 23, 'PERSON'), (24, 38, 'DESIGNATION')]}), ('j. j. abrams directed by', {'entities': [(0, 12, 'PERSON'), (13, 24, 'DESIGNATION')]}), ('kathleen kennedy j. j. abrams,michelle rejwan,produced by', {'entities': [(0, 16, 'PERSON'), (17, 57, 'DESIGNATION')]}), ('chris terrio screenplay by', {'entities': [(0, 12, 'PERSON'), (13, 26, 'DESIGNATION')]}), ('j. j. abrams screenplay by', {'entities': [(0, 12, 'PERSON'), (13, 26, 'DESIGNATION')]}), ('derek connolly story by', {'entities': [(0, 14, 'PERSON'), (15, 23, 'DESIGNATION')]}), ('colin trevorrow story by', {'entities': [(0, 15, 'PERSON'), (16, 24, 'DESIGNATION')]}), ('j. j. abrams chris terrio story by', {'entities': [(0, 12, 'PERSON'), (13, 25, 'PERSON'), (26, 34, 'DESIGNATION')]}), ('george lucas based on characters by', {'entities': [(0, 13, 'PERSON'), (14, 36, 'DESIGNATION')]}), ('john williams music by', {'entities': [(0, 13, 'PERSON'), (14, 22, 'DESIGNATION')]}), ('dan mindel cinematography', {'entities': [(0, 10, 'PERSON'), (11, 25, 'DESIGNATION')]}), ('maryann brandon edited by', {'entities': [(0, 15, 'PERSON'), (16, 25, 'DESIGNATION')]}), ('stefan grube edited by', {'entities': [(0, 12, 'PERSON'), (13, 22, 'DESIGNATION')]}), ('lucasfilm ltd. production company', {'entities': [(0, 14, 'PERSON'), (15, 33, 'DESIGNATION')]}), ('bad robot productions production company', {'entities': [(0, 21, 'PERSON'), (22, 40, 'DESIGNATION')]}), ('walt disney studios distributed by', {'entities': [(0, 19, 'PERSON'), (20, 34, 'DESIGNATION')]}), ('motion pictures distributed by', {'entities': [(0, 15, 'PERSON'), (16, 30, 'DESIGNATION')]}), ('john paesano music by', {'entities': [(0, 12, 'PERSON'), (13, 21, 'DESIGNATION')]}), ('james powers unit proouction manager', {'entities': [(0, 12, 'PERSON'), (13, 36, 'DESIGNATION')]}), ('vincent lascoumes first assistant director', {'entities': [(0, 17, 'PERSON'), (18, 42, 'DESIGNATION')]}), ('jeff jj authors first assistant director', {'entities': [(0, 15, 'PERSON'), (16, 40, 'DESIGNATION')]}), ('nadia brand second assistant directors', {'entities': [(0, 11, 'PERSON'), (12, 38, 'DESIGNATION')]}), ('joy hoes second assistant directors', {'entities': [(0, 8, 'PERSON'), (9, 35, 'DESIGNATION')]}), ('david klohn second assistant directors', {'entities': [(0, 11, 'PERSON'), (12, 38, 'DESIGNATION')]}), ('greg kriek stunt coordinator', {'entities': [(0, 10, 'PERSON'), (11, 28, 'DESIGNATION')]}), ('david butler assistant stunt coordinator', {'entities': [(0, 12, 'PERSON'), (13, 40, 'DESIGNATION')]}), ('liza scholtz fight choreographer', {'entities': [(0, 12, 'PERSON'), (13, 32, 'DESIGNATION')]}), ('susan mcnamara unit production manager', {'entities': [(0, 14, 'PERSON'), (15, 38, 'DESIGNATION')]}), ('david valdes unit production manager', {'entities': [(0, 12, 'PERSON'), (13, 36, 'DESIGNATION')]}), ('brian bettwy first assistant director', {'entities': [(0, 12, 'PERSON'), (13, 37, 'DESIGNATION')]}), ('david vincent rimer second assistant director', {'entities': [(0, 19, 'PERSON'), (20, 45, 'DESIGNATION')]}), ('dawn robinette production accountant', {'entities': [(0, 14, 'PERSON'), (15, 36, 'DESIGNATION')]}), ('garrett warren stunt coordinator', {'entities': [(0, 14, 'PERSON'), (15, 32, 'DESIGNATION')]}), ('michael cozens weta animation supervisor', {'entities': [(0, 14, 'PERSON'), (15, 40, 'DESIGNATION')]}), ('caylah eddleblute co-production designer', {'entities': [(0, 17, 'PERSON'), (18, 40, 'DESIGNATION')]}), ('a.todd holland supervising art director', {'entities': [(0, 14, 'PERSON'), (15, 39, 'DESIGNATION')]}), ('tom proper post production supervisor', {'entities': [(0, 10, 'PERSON'), (11, 37, 'DESIGNATION')]}), ('jason gaudio supervising first assistant editor', {'entities': [(0, 12, 'PERSON'), (13, 47, 'DESIGNATION')]}), ('steve brown assistant stunt coordinator', {'entities': [(0, 11, 'PERSON'), (12, 39, 'DESIGNATION')]}), ('steve brown fight choreographer', {'entities': [(0, 11, 'PERSON'), (12, 31, 'DESIGNATION')]}), ('leslie mcdonald art directors', {'entities': [(0, 15, 'PERSON'), (16, 29, 'DESIGNATION')]}), ('paul alix assistant art director', {'entities': [(0, 9, 'PERSON'), (10, 32, 'DESIGNATION')]}), ('jim mitchell storyboard artists', {'entities': [(0, 12, 'PERSON'), (13, 31, 'DESIGNATION')]}), ('john berger set designers', {'entities': [(0, 11, 'PERSON'), (12, 25, 'DESIGNATION')]}), ('natalie angel production supervisor', {'entities': [(0, 13, 'PERSON'), (14, 35, 'DESIGNATION')]}), ('cynthia streit production coordinator', {'entities': [(0, 14, 'PERSON'), (15, 37, 'DESIGNATION')]})
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model="en_core_web_md", new_model_name="designation", output_dir="/home/nithing/PycharmProjects/New1/model_dir", n_iter=1000):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")

    ner.add_label(LABEL)  # add new entity label to entity recognizer
    # Adding extraneous labels shouldn't mess anything up
    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            batches = minibatch(TRAIN_DATA, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.35, losses=losses)
            print("Losses", losses)

    # test the trained model
    test_text = "produced by Christopher from London?"
    doc = nlp(test_text)
    print("Entities in '%s'" % test_text)
    for ent in doc.ents:
        print(ent.label_, ent.text)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
        doc2 = nlp2(test_text)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == "__main__":
    plac.call(main)