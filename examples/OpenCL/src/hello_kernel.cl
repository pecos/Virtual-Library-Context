kernel void say_hello( global char *str){
      int id = get_global_id(0);
       switch (id) {
           case 0:
               str[id] = 'H';
               break;
           case 1:
               str[id] = 'e';
               break;
           case 2:
               str[id] = 'l';
               break;
           case 3:
               str[id] = 'l';
               break;
           case 4:
               str[id] = 'o';
               break;
           case 5:
               str[id] = ' ';
               break;
           case 6:
               str[id] = 'W';
               break;
           case 7:
               str[id] = 'o';
               break;
           case 8:
               str[id] = 'r';
               break;
           case 9:
               str[id] = 'l';
               break;
           case 10:
               str[id] = 'd';
               break;
           case 11:
               str[id] = '\n';
               break;
           case 12:
               str[id] = '\0';
               break;

           default:
               break;
       }
}