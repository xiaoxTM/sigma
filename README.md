# Software Integration Group Machine Armory
![sigma](logos/sigma.png)

sigma is short for `S`oftware `I`ntegration `G`roup `M`achine-Learning `A`rmory. Unlike `Keras` which is class based, `sigma` is a functional fashion framework. That is, all layers are functions rather than classes.

# Version explanation
`sigma` version consists of three parts:
  - major version indicator
    - increases after great changes
  - minor version indicator
    - increases after adding new features
  - update indicator
    - increases after each reported bugs fixing
  - state version indicator
    - 0 : releasable (mainly clean code up)
    - odd : developing state
    - even (except for 0) : testing state
  - pid version indicator
    - used for multi-person parallelly developments
      (that is, used for distinguishing developers)

# Developing progress

```
                              devel
                                ^
                                |
     devel     developer-0  /-------\           /-------\
      ^^      /----------->|x.x.y.o.0|<------->|x.x.y.e.0| => test -------|
      ||     /              \-------/           \-------/                 |
   /-------\/  developer-1  /-------\           /-------\                 |
  | x.x.y.o |------------->|x.x.y.o.1|<------->|x.x.y.e.1| => test ------>|
   \-------/\               \-------/           \-------/                 |
       ^     \ developer-2  /-------\           /-------\                 |
       |      \----------->|x.x.y.o.2|<------->|x.x.y.e.2| => test ------>|
       |                    \-------/           \-------/                 |
       |                                                                  v
       |bug reports                                                   /-------\           /-------\
       |                                         integration test <= | x.x.y.e |<------->|x.x.y+1.0| => stable
       |                                                              \-------/           \-------/
       |                                                                                      |
       |                                                                                      v
   /------\                                                                               /-------\
  | master |---------------------------------------------------------------------------->|  master |
   \------/                                merge x.x.y.0                                  \-------/
```
