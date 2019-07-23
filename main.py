from tkinter import *

LARGE_FONT = ("Verdana", 12)


class WoundRepairTool(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master)

        self.master = master

        self.init_window()


    def init_window(self):

        self.master.title("Wound Repair Tool")
        self.pack(fill=BOTH, expand=1)

        quit_button = Button(self, text='Cancel', command=lambda: exit())
        quit_button.place(x=0, y=0)

        self.size()




if __name__ == '__main__':

    # the root window
    root = Tk()
    root.geometry('400x400')
    app = WoundRepairTool(root)
    root.mainloop()

    # root.withdraw()


    # filenames = askopenfilenames()
    # root.update()
    # print(filenames)



