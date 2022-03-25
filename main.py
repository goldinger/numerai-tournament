import inspect
from models import lgbm


options = [
    ('LightGBM Regressor', lgbm.run),
    ('Exit', quit)
]

def menu() -> str:
    print('Select a model to run:')
    for i, (label, func) in enumerate(options):
        print(f"{i+1}. {label}")
    choice = 'error'
    return input("Your choice : ")
    


def main():
    choice = 'choice'
    while not (choice.isdigit() and 0 < int(choice) <= len(options)):
        choice = menu()
    choice = int(choice)
    func = options[choice - 1][1]
    if func == quit:
        print('Good Bye !')
        quit()

    print("Excellent choice !")
    args = input("Any arguments ? (y/n) :")
    kwargs = {}
    if args.lower() == 'y':
        
        for parameter in inspect.signature(func).parameters.values():
            text = parameter.name
            if parameter.annotation != parameter.empty:
                if parameter.annotation not in [str, int]:
                    raise NotImplementedError(f"{parameter.annotation.__name__} arguments are not supported")
                text += f" ({parameter.annotation.__name__})"
            if parameter.default != parameter.empty:
                text += f" ({parameter.default})"
            
            text += ':'
            value = input(text)
            if value == "" and parameter.default != parameter.empty:
                value = parameter.default
            elif parameter.annotation == parameter.empty:
                type_name = input("    Type (str, int): ")
                value = getattr(__builtins__, type_name)(value)
            else:
                
                value = parameter.annotation(value)
            
            kwargs[parameter.name] = value
        
    func(**kwargs)

if __name__ == '__main__':
    main()