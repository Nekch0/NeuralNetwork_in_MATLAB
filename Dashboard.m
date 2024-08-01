function varargout = Dashboard(varargin)
    gui_Singleton = 1;
    gui_State = struct('gui_Name',       mfilename, ...
                       'gui_Singleton',  gui_Singleton, ...
                       'gui_OpeningFcn', @Dashboard_OpeningFcn, ...
                       'gui_OutputFcn',  @Dashboard_OutputFcn, ...
                       'gui_LayoutFcn',  [] , ...
                       'gui_Callback',   []);
    if nargin && ischar(varargin{1})
        gui_State.gui_Callback = str2func(varargin{1});
    end
    if nargout
        [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
    else
        gui_mainfcn(gui_State, varargin{:});
    end
end

function Dashboard_OpeningFcn(hObject, eventdata, handles, varargin)
    handles.output = hObject;
    guidata(hObject, handles);
end

function varargout = Dashboard_OutputFcn(hObject, eventdata, handles) 
    varargout{1} = handles.output;
end

% ==============================================================================

function Dashboard_WindowKeyPressFcn(hObject, eventdata, handles)
end
