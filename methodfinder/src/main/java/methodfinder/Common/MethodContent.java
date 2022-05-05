package methodfinder.Common;

import com.github.javaparser.ast.body.MethodDeclaration;

public class MethodContent {
	private MethodDeclaration node;
	private String name;
	private long length;

	public MethodContent(MethodDeclaration node, String name, long length) {
		this.node = node;
		this.name = name;
		this.length = length;
	}

	public MethodDeclaration getNode() {
		return node;
	}

	public String getName() {
		return name;
	}

	public long getLength() {
		return length;
	}

}
