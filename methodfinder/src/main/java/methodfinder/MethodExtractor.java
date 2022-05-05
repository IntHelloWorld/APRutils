package methodfinder;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import com.github.javaparser.ast.CompilationUnit;

import methodfinder.Common.CommandLineValues;
import methodfinder.Common.Common;
import methodfinder.Common.MethodContent;
import methodfinder.Visitors.FunctionVisitor;

public class MethodExtractor {

    public MethodExtractor(CommandLineValues commandLineValues) {
    }

    public ArrayList<MethodContent> extractFeatures(String code, Path filePath, List<String> targetMethodNames)
            throws IOException {
        CompilationUnit compilationUnit = Common.parseFileWithRetries(code);
        FunctionVisitor functionVisitor = new FunctionVisitor(targetMethodNames);
        functionVisitor.visit(compilationUnit, null);
        return functionVisitor.getMethodContents();
    }
}
