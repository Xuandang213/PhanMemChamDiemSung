"""Place child windows centrally, keeping their content on screen."""


def center_window(window,parent=None):
    window.update_idletasks()
    parent=parent or window.master
    sw,sh=window.winfo_screenwidth(),window.winfo_screenheight()
    width=min(max(window.winfo_width(),window.winfo_reqwidth()),sw-40)
    height=min(max(window.winfo_height(),window.winfo_reqheight()),sh-90)
    if parent is not None and parent.winfo_viewable():
        parent=parent.winfo_toplevel()
        cx=parent.winfo_rootx()+parent.winfo_width()/2
        cy=parent.winfo_rooty()+parent.winfo_height()/2
    else:cx,cy=sw/2,sh/2
    x=max(10,min(round(cx-width/2),sw-width-10))
    y=max(30,min(round(cy-height/2),sh-height-60))
    window.geometry(f'{width}x{height}+{x}+{y}')
