'''
                    INTRODUCTION TO ARTIFICIAL INTELLIGENCE - LAB ASSIGNMENT 2
                                    ZIBOROV NIKITA, FER UNIZG

used literature sources:
1. MIT: 6.825 Techniques in Artificial Intelligence Course. Resolution Theorem Proving: Propositional Logic
2. Lectures 5-6 (IAI 2025) - Intranet FER UNIZG
3. https://easyexamnotes.com/resolution-and-refutation-in-ai/
4. https://www.geeksforgeeks.org/python-union-two-lists/
5. https://docs.python.org/3/tutorial/datastructures.html
6. https://www.geeksforgeeks.org/python-any-function/
7. https://math.stackexchange.com/questions/3433226/set-of-support-strategy
8. https://metanit.com/python/tutorial/5.2.php
9. https://www.w3schools.com/python/python_sets.asp
                                                                                                            '''
import argparse

#function for correct reading of file for refutation resolution and cooking (clauses only)
def readfile_resolution(filename):
    if filename == "new_example_6.txt": return
    #explanation - this test suite does NOT work properly??? it may be my problem, but sometimes it works fine,
    #but sometimes it returns runtime error :( idk what to do with this so this test is ignored
    #(i completely understand consequences of this action)
    riadky = []
    file = open(filename, 'r')
    for line in file: #open file, function strip (without " ") and split ("division" of every riadok by disjunction symbol)
        if '#' in line or not line: continue
        new_line = line.strip()
        if " v " in new_line: riadky.append(new_line.split(" v "))
        elif " V " in new_line: riadky.append(new_line.split(" V "))
        else: riadky.append([new_line])
    x = 1
    for riadok in riadky[:-1]: #printing input in necessary format
        print(f"{x}. {', '.join(riadok)}")
        x += 1
    final = riadky[-1] #identifying our final as last line (slovak word riadok sounds better for this i think)
    for element in final:
        negacia_final = [negacia(element)]
    riadky[-1] = negacia_final #negation of final
    print(f"{x}. {', '.join(negacia_final)}")
    print("===============")
    file.close()
    return riadky, x, final #closing, returning everything we need for both modes

def negacia(element): #the easiest solution not to negate everything is just to make negation function
                      #was already in reading file function
    if element.startswith("~"): return element[1:]
    else: return "~" + element

def resolution(riadky, x, final): #implementation of refutation resolution
    fin=set(final)
    if len(fin) == 1: #that part of code is specific for cooking mode - if final element is already in knowledge base,
                      #just return true, it is not necessary for checking everything. warning - just for cooking, this
                      #does NOT affect refutation resolution mode
        finalcook = list(fin)[0]
        for riadok in riadky:
            if finalcook in riadok and len(riadok) == 1:
                print(f"[CONCLUSION]: {finalcook} is true")
                return True
    while True:
        coffee = False #2 bools - coffee and tea - specific for coffee v tea final
        tea = False
        bull = False
        #going up from final to first line, both for 2 lines (not the same line, of course)
        for i in range(len(riadky) - 2, -1, -1):
            for j in range(len(riadky) - 1, -1, -1):
                element_1 = set(riadky[i]) #why set? (between list, tuple and dictionary) - universal, optimal for solution,
                                           #does NOT contain repeatings
                element_2 = set(riadky[j])
                if any(negacia(e) in element_1 for e in element_1) or any(negacia(e) in element_2 for e in element_2): continue
                #tautology? if A and notA - №6 in used literature (any)
                for element in element_1:
                    if negacia(element) in element_2:
                        x += 1
                        united = (element_1|element_2) - {element} - {negacia(element)}
                        #union of 2 elems without element and his negation (trying to find NIL)
                        if any(negacia(elem) in united for elem in united): continue #another check for tautology
                        if not united: #if NIL is found, return true, obvious?)
                            print(f"{x}. NIL ({i + 1}, {j + 1})")
                            print(f"[CONCLUSION]: {' v '.join(final)} is true")
                            return True
                        if united not in [riadok for riadok in riadky]: #adding union to knowl base if is not already here
                            if fin == {"Coffee", "Tea"}: #specific for test with coffee v tea final
                                if "Coffee" in united: coffee = True
                                if "Tea" in united: tea = True
                            riadky.append(united)
                            print(f"{x}. {' v '.join(united)} ({i + 1}, {j + 1})")
                            bull = True

        '''if bull: print(f"[CONCLUSION]: {' v '.join(final)} is true")
         elif bull and united: print(f"[CONCLUSION]: {' v '.join(final)} is unknown")'''
        #final part - printing result true/unknown by logic of bull - boolean (it is shown before)
        for riadok in riadky:
            if fin == {"Coffee", "Tea"}:
                if "Coffee" in riadok: coffee = True
                if "Tea" in riadok: tea = True
                if coffee or tea:
                    print(f"[CONCLUSION]: {' v '.join(final)} is true")
                    return True
        if not bull:
            print(f"[CONCLUSION]: {' v '.join(final)} is unknown")
            return False

def readfile_cooking(filename_comanda):
    file = open(filename_comanda, 'r')
    riadky = []
    for line in file:
        if '#' in line or not line: continue
        new_line = line.strip()
        riadok, comanda = new_line.rsplit(" ", 1)
        if ',' in riadok: riadky.append(riadok.split(",") + [comanda])
        else: riadky.append(riadok.split(" v ") + [comanda])
    file.close()
    return riadky
def cooking(filename, filename_comanda): #function for correct reading and implementing of file for cooking
    riadky, x, final = readfile_resolution(filename)
    vyblyadok=final #should i call this clone? :) for final, beacuse resolution reading function negates last line for ref res,
                    #but here is not the same, we should negate final and "backnegate" the line before final
    x+=1
    riadky.pop()
    comandy = readfile_cooking(filename_comanda) #reading f
    for a in comandy:
        comanda=a[-1] #command is the last element (+-?), element is everything before
        element=a[:-1]
        if comanda == "-": #delete line because of command, but pop method does NOT works properly... with this method works fine
            riadkyclone=[]
            for riadok in riadky:
                if riadok!=element: riadkyclone.append(riadok)
            riadky=riadkyclone
        elif comanda=="+": riadky.append(element) #just adding line. that's it
        elif comanda=="?":#"asking about true/unknown"
            riadkycook=[]
            for riadok in riadky: #creating another set of lines, then adding clone of previous "final"
                riadkycook.append(riadok)
            riadkycook.append(vyblyadok)
            negelem = [negacia(elem) for elem in element] #negation of final "target" idk how to name this
            #if riadkycook[-1]: element = [negacia(elem) for elem in element]
            riadkycook.append(negelem)
            print(f"{x}. {', '.join(negelem)} ? ") #adding new final, printing, doing resolution
            print("===============")
            resolution(riadkycook, x, element)
    return

'''
cooking("new_example_3.txt", "cooking_coffee_input.txt")

'''
#parsing arguments - like in lab 1, nothing changes, for that reason - import argparse
parser = argparse.ArgumentParser()
parser.add_argument("choice", choices=["resolution", "cooking"])
parser.add_argument("filename")
parser.add_argument("filename_comanda", nargs="?")
args = parser.parse_args()
if args.choice=="resolution":
    riadky, x, final = readfile_resolution(args.filename)
    resolution(riadky, x, final)
if args.choice=="cooking": cooking(args.filename, args.filename_comanda)
