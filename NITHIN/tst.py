#!/usr/bin/env python
# coding: utf8
"""Example of training an additional entity type

This script shows how to add a new entity type to an existing pretrained NER
model. To keep the example short and simple, only four sentences are provided
as examples. In practice, you'll need many more — a few hundred would be a
good start. You will also likely need to mix in examples of other entity
types, which might be obtained by running the entity recognizer over unlabelled
sentences, and adding their annotations to the training set.

The actual training is performed by looping over the examples, and calling
`nlp.entity.update()`. The `update()` method steps through the words of the
input. At each word, it makes a prediction. It then consults the annotations
provided on the GoldParse instance, to see whether it was right. If it was
wrong, it adjusts its weights so that the correct action will score higher
next time.

After training your model, you can save it to a directory. We recommend
wrapping models as Python packages, for ease of deployment.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.1.0+
Last tested with: v2.1.0
"""
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
    (' George Gallo,Based on by', {'entities': [(0, 13, 'PERSON'), (14, 25, 'DESIGNATION')]}),
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
    (' George Lucas,Based on Characters by', {'entities': [(0, 13, 'PERSON'), (14, 36, 'DESIGNATION')]}),
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
    ('DAN EVANS FARKAS,Music Editor', {'entities': [(0, 16, 'PERSON'), (17, 29, 'DESIGNATION')]})
]


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model="en_core_web_md", new_model_name="DESIGNATION", output_dir="model_dir", n_iter=30):
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
    ner.add_label("VEGETABLE")
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
    test_text = "Do you like horses?"
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
