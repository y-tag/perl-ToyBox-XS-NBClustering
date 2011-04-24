use 5.008000;
#use ExtUtils::MakeMaker;
use inc::Module::Install;

name          'ToyBox-XS-NBClustering';
all_from      'lib/ToyBox/XS/NBClustering.pm';
license       'MIT';

$CC= '/usr/bin/g++';

# See lib/ExtUtils/MakeMaker.pm for details of how to influence
# the contents of the Makefile that is written.
WriteMakefile(
    NAME              => 'ToyBox::XS::NBClustering',
    VERSION_FROM      => 'lib/ToyBox/XS/NBClustering.pm', # finds $VERSION
    PREREQ_PM         => {}, # e.g., Module::Name => 1.1
    ($] >= 5.005 ?     ## Add these new keywords supported since 5.005
      (ABSTRACT_FROM  => 'lib/ToyBox/XS/NBClustering.pm', # retrieve abstract from module
       AUTHOR         => 'TAGAMI Yukihiro <tagami@>') : ()),
    LIBS              => ['-lm'], # e.g., '-lm'
    DEFINE            => '', # e.g., '-DHAVE_SOMETHING'
    CC                => $CC,
    LD                => '$(CC)',
    INC               => '-I.', # e.g., '-I. -I/usr/include/other'
	# Un-comment this if you add C files to link with later:
    # OBJECT            => '$(O_FILES)', # link all the C files too
    XSOPT             => '-C++',
    TYPEMAPS          => ['perlobject.map'],
);
